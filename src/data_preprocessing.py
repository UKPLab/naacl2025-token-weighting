import os
import pickle
import more_itertools
import numpy as np
import torch

from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, default_data_collator
from typing import Optional
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from logger_setup import logger
from run_settings import RunSettings
from utils import set_random_state, get_model
from trainer import calculate_base_loss

def get_data(settings: RunSettings, split:str="training"):
    """
    Fetches and preprocesses the training data for a given configuration.

    This function uses the provided settings and split to initialize a data
    preprocessor. It attempts to load precompiled training data from disk. If the
    precompiled data is unavailable or causes any exception, the function retrieves
    the raw data, preprocesses it, and saves it back to disk before returning the
    result. This process ensures consistent data handling and reproducibility.

    :param settings: Configuration settings including seed and data paths.
    :type settings: Any
    :param split: The specific data split to process (e.g., "training").
    :type split: str
    :return: The loaded and potentially preprocessed training dataset.
    :rtype: Any
    """
    data_preprocessor = DataPreprocessor(split=split, settings=settings)

    logger.info(f"Loading training data from {data_preprocessor.datapath}")
    set_random_state(settings.seed)
    try:
        train_data = load_from_disk(data_preprocessor.datapath)
    except Exception as e:
        logger.info(f"Error loading precompiled training dataset, loading from disk instead")
        train_data = data_preprocessor.fetch_data_from_file("training")
        train_data.save_to_disk(data_preprocessor.datapath)
    logger.info("Train data was successfully loaded")

    return train_data, data_preprocessor

def precompute_weights(settings):
    """
    Precomputes and saves the weights of the scoring model using preprocessed data. The function evaluates
    the given model in a no-gradient context and performs computation on batches of data, storing the
    results in a designated directory. It ensures the computation environment is configured
    appropriately (e.g., device setup) and saves results in a compressed format for efficient storage.

    :param settings: Configuration settings that include batch size, paths for preprocessed data,
        scoring model, custom loss parameters, tokenizer, and minibatch size.
    :type settings: Any
    :return: Path where the precomputed weights are saved.
    :rtype: str
    """
    data, data_preprocessor = get_data(settings)
    model = get_model(settings)

    save_path = os.path.join(f"{data_preprocessor.PATH_CHUNKED_DATA}",
                             f"precomputed_weights_{data_preprocessor.scoring_model}")
    os.makedirs(save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator()
    dataloader = DataLoader(data,
                            collate_fn=default_data_collator,
                            batch_size=settings.batch_size, shuffle=False)
    for param in model.parameters():
        param.requires_grad = False
    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()
    batch_no = 0
    for big_batch in tqdm(dataloader):
        batch = big_batch["input_ids"].to(device)
        device_no = batch.get_device()

        with torch.inference_mode():
            with autocast(dtype=torch.bfloat16):
                cross_entropy = calculate_base_loss(model, batch, settings.custom_loss.base_length,
                                                    settings.custom_loss.base_stride,
                                                    tokenizer=settings.tokenizer,
                                                    use_first_chunk=True,
                                                    minibatch_size=settings.scoring_minibatch
                                                    )
        batch = batch.detach().cpu().numpy()
        cross_entropy = cross_entropy.detach().cpu().numpy()

        file_path = os.path.join(save_path, f"precomputed_weights_{batch_no}_{device_no}.npz")
        if os.path.exists(file_path):
            logger.info(f"File {batch_no} {device_no} already exists")
        else:
            np.savez_compressed(file_path,
                                input_ids=batch, cross_entropy=cross_entropy)

        logger.info(f"Cross Entropy: {cross_entropy}")
        batch_no += 1
    return save_path

def preprocess_function(token_ids, chunk_factor: int, bos_token_id: Optional[int] = None):
    """
    This method reshapes long sequences into ones that are chunk_factor times shorter. It ensures that the shorter
    sequences are also prepended with BOS token, if given.
    :param token_ids:
    :param chunk_factor:
    :param bos_token_id:
    :return:
    """
    splits = np.array_split(token_ids, chunk_factor, axis=1)
    if bos_token_id is not None and chunk_factor > 1:
        splits = [np.concatenate(
            (np.ones((len(token_ids), 1), np.int32) * bos_token_id, split), axis=1)[:, :-1] if
                  split_id > 0 else split for split_id, split in enumerate(splits)]
    token_ids = np.concatenate(splits, axis=0)

    if bos_token_id is not None:
        assert np.array_equal(token_ids[:, 0], np.ones_like(token_ids[:, 0]) * bos_token_id)
        assert bos_token_id not in list(token_ids[:, 1])

    return token_ids

class DataPreprocessor:
    def __init__(self, PATH_RAW_DATA: str = None, tokenizer=None, dataset: str = "pg19", split: str = "training",
                 chunk_size: int = 32768, stride: int = 32768,
                 discard_last: bool = True, discard_first: bool = False, scoring_model: str = "",
                 max_chunk_factor: int = 16, PATH_CHUNKED_DATA: str = None,
                 no_train: int = None, train_load_size=1000,
                 seed: int = 42, use_frozen_base: bool = False,
                 num_train_sequences: int = None,
                 settings: RunSettings = None):
        """

        :param dataset: dataset name
        :param split: training or validation (naming depends on hf dataset)
        :param chunk_size: number of tokens per sequence
        :param stride: how many tokens there are between the beginning of sequences. if stride < chunk_size we have a token overlap between sequences
        :param discard_last: if the last sequence should be discarded if it is shorter than chunk_size
        :param discard_first: if the first sequence of a document should be discarded
        :param scoring_model: short name for the model, with which the frozen losses are computed
        :param max_chunk_factor: if we need a dataset of chunk_size n, we can create it by using a dataset of
        chunk_size k*n and splitting the sequences (provided we created it at some point in the past)
        max_chunk_factor is the maximal value of k we search for
        :param PATH_CHUNKED_DATA: path to the tokenized and chunked data. Will use data from there and don't tokenize freshly
        :param no_train: number of training examples to be processed. None uses the full dataset
        :param train_load_size: how many documents are tokenized at once
        :param PATH_RAW_DATA: directory where downloaded PG19 data is or where it should be downloaded to (will be created if non-existent)
        :param tokenizer: the tokenizer
        :param seed: random seed
        :param use_frozen_base: if yes, we will use a dataset with a column containing frozen base losses
        :param num_train_sequences: how many sequences the created hf dataset should contain
        :param settings: if not None, the DataPreprocessor object will inherit all attributes of the same name from the RunSettings object
        """

        if dataset is not None and "-" in dataset:
            self.dataset, self.data_subset = dataset.split("-")
        else:
            self.dataset = dataset
            self.data_subset = None

        self.split = split
        self.chunk_size = chunk_size
        self.stride = stride
        self.discard_last = discard_last
        self.discard_first = discard_first

        self.no_train = no_train

        self.train_load_size = train_load_size

        self.PATH_RAW_DATA = PATH_RAW_DATA

        if isinstance(tokenizer, str) and tokenizer == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        else:
            self.tokenizer = tokenizer

        self.seed = seed
        self.use_frozen_base = use_frozen_base

        self.num_train_sequences = num_train_sequences

        self.max_chunk_factor = max_chunk_factor

        self.PATH_CHUNKED_DATA = PATH_CHUNKED_DATA


        self.scoring_model = "3.0_8B" if scoring_model == "" else scoring_model
        assert self.scoring_model in ["3.0_8B", "3.1_8B", "3.2_3B", "3.2_1B"]

        if settings is not None:
            for attr in dir(self):
                if not attr.startswith("__") and hasattr(settings, attr) and attr is not None:
                    setattr(self, attr, getattr(settings, attr))
        else:
            if self.tokenizer is None or self.PATH_RAW_DATA is None:
                raise ValueError(
                    "tokenizer and PATH_RAW_DATA could not be inferred because no settings were given. Please set them explicitly.")

        os.makedirs(self.PATH_CHUNKED_DATA, exist_ok=True)

        frozen_indicator = "frozen" if self.use_frozen_base else ""
        self.scoring_model = "3.0_8B" if scoring_model == "" else scoring_model

        if self.use_frozen_base:
            frozen_indicator = self.scoring_model
        else:
            frozen_indicator = False

        if self.PATH_CHUNKED_DATA is not None:
            self.datapath = os.path.join(self.PATH_CHUNKED_DATA,
                                         f"{self.dataset}_{self.split}_chunk_size_{self.chunk_size}_stride_{self.stride}_last_{not self.discard_last}_first_{not self.discard_first}_frozen_{frozen_indicator}.hf")
        else:
            self.PATH_CHUNKED_DATA = None

        if "-" in self.dataset:
            self.dataset, self.data_subset = self.dataset.split("-")

    def save_locally_hf(self, dataset):
        dataset.save_to_disk(self.datapath)
        logger.info(f"Dataset was saved at {self.datapath}")
        return self.datapath


    def fetch_data_from_file(self, split=None):
        """
        Fetches data from a file based on the specified data split. The data is fetched and
        prepared for further usage. If the data files are not found, the method attempts
        to consolidate or re-chunk the data before raising an error in case of failure.

        This is a high-level method that delegates operations to other methods like
        `get_all_data`, `consolidate_data`, `chunk_data_anew`, and `prepare_data`. It is
        designed to handle dynamic preparation and retrieval of data by managing edge
        cases, including missing or corrupted files.

        :param split: The data split to be fetched. Expected values are "training" or
                      "validation". If no split is provided, defaults to the object's
                      defined `split` attribute.
        :type split: str, optional
        :return: Processed data based on the specified split and other configurations.
        :rtype: any
        """
        set_random_state(self.seed)
        if split is None:
            split = self.split

        if split == "training":
            no_samples = self.no_train
        else:
            raise ValueError("split has to be either training or validation")

        try:
            data_array, chunk_factor = self.get_all_data(shard="full", shuffled=True)
        except (FileNotFoundError, TypeError):
            try:
                self.consolidate_data()
                data_array, chunk_factor = self.get_all_data(shard="full", shuffled=True)
            except (FileNotFoundError, TypeError):
                self.chunk_data_anew()
                try:
                    self.consolidate_data()
                    data_array, chunk_factor = self.get_all_data(shard="full", shuffled=True)
                except FileNotFoundError:
                    raise FileNotFoundError(f"no data found for {split}")

        return self.prepare_data(data_array, chunk_factor)

    def prepare_data(self, data, chunk_factor):
        data_array = preprocess_function(data, chunk_factor=chunk_factor,
                                         bos_token_id=self.tokenizer.bos_token_id
                                         if chunk_factor > 1 and self.tokenizer.bos_token is not None
                                         else None)
        logger.info(f"final shape: {data_array.shape}")

        if self.use_frozen_base and self.split == "training":
            cross_entropy_array = np.zeros((data_array.shape[0], data_array.shape[1] - 1))
            entropy_array = np.zeros((data_array.shape[0], data_array.shape[1] - 1))
            logger.info(f"cross_entropy_array shape: {cross_entropy_array.shape}")
            logger.info(f"entropy_array shape: {entropy_array.shape}")
            with open(self.use_frozen_base, 'rb') as fp:
                precomputed_weights = pickle.load(fp)
            logger.info("after load")

            for i in range(len(data_array)):
                key = tuple(data_array[i, :])
                if key in precomputed_weights.keys():
                    cross_entropy_array[i, :] = precomputed_weights[key][0]
                    entropy_array[i, :] = precomputed_weights[key][1]
                    logger.info(f"after input_ids {i}")
                else:
                    assert False, f"Key {key} not found in precomputed_weights"
            cross_entropy_array = torch.tensor(cross_entropy_array, dtype=torch.float32)
            entropy_array = torch.tensor(entropy_array, dtype=torch.float32)

        if self.num_train_sequences is not None and self.num_train_sequences >= 0:
            no_datapoints_considered = min(len(data_array), self.num_train_sequences)
        else:
            no_datapoints_considered = len(data_array)

        data_array = torch.tensor(data_array, dtype=torch.long)

        def frozen_gen():
            for j in range(no_datapoints_considered):
                yield {"input_ids": data_array[j, :],
                       "cross_entropy": cross_entropy_array[j, :],
                       "entropy": entropy_array[j, :]}

        def gen():
            for j in range(no_datapoints_considered):
                yield {"input_ids": data_array[j, :]}

        if self.use_frozen_base and self.split == "training":
            set_random_state(self.seed)
            train_data = Dataset.from_generator(frozen_gen)
        else:
            set_random_state(self.seed)
            train_data = Dataset.from_generator(gen)
        return train_data

    def get_all_data(self, shard="full", shuffled=False):
        """
        Fetches and returns all data for the specified shard and shuffle option.

        This method attempts to load pre-computed data from a cache. The method will
        iterate through different chunk factors until a valid cached file is found or
        exceeds the maximum chunk factor. If an appropriate cache file is located, it
        loads the data and returns it along with the chunk factor. If no cache file
        is found, a `FileNotFoundError` is raised.

        :param shard: Specifies the subset of the data to load. Default is "full".
                      Valid options may depend on the specific dataset configuration.
        :type shard: str
        :param shuffled: Indicates whether to load the shuffled version of the data.
                         If set to True, the method appends a "_shuffled" suffix to
                         the file path. Default is False.
        :type shuffled: bool
        :return: A tuple containing the loaded data and the corresponding chunk factor.
        :rtype: tuple
        :raises FileNotFoundError: If no appropriate cached data file is found.
        """
        stride = self.stride
        chunk_size = self.chunk_size

        shuffle_string = "_shuffled" if shuffled else ""

        cache_available = False
        chunk_factor = 1
        while chunk_factor <= self.max_chunk_factor:
            path = os.path.join(self.PATH_CHUNKED_DATA,
                                f"{self.dataset}_{self.split}_{shard}_chunk_size_{chunk_factor * chunk_size}_stride_{stride}_last_{not self.discard_last}_first_{not self.discard_first}{shuffle_string}.npy")
            try:
                logger.info(
                    f"trying to load from {path}")
                cached_data = np.load(path)
                logger.info(f"used cached {self.split} data with factor {chunk_factor}")
                cache_available = True
                break
            except FileNotFoundError:
                chunk_factor *= 2
        if cache_available:
            return cached_data, chunk_factor
        else:
            raise FileNotFoundError(f"no data found for {self.split} and shard {shard}")

    def consolidate_data(self):
        """
        Consolidates dataset batches by loading, concatenating, and saving them as a single file.
        This method processes training dataset batches and handles caching to enable efficient data consolidation.
        The output dataset is shuffled before saving and ensures reproducibility by adhering to specified parameters.

        :param self: The instance of the class that manages dataset attributes and configurations.
        :raises ValueError: If the split is not "training".
        :raises FileNotFoundError: If the required batch data for the specified shard is not found.
        :return: None
        """
        batch_begin = 0
        if self.split == "training":
            no_of_books = self.train_load_size

            if self.no_train is None:
                dataset = load_dataset(self.dataset, trust_remote_code=True,
                                       split=(self.split if self.split == "validation" else "train"),
                                       num_proc=8, cache_dir=self.PATH_RAW_DATA)
                self.no_train = len(dataset["text"])

            dataset_size = self.no_train
        else:
            raise ValueError("split has to be either training or validation")
        all_data = []
        while batch_begin < dataset_size:
            batch_end = min(batch_begin + no_of_books, dataset_size)

            shard = f"{batch_begin}_{batch_end}"
            try:
                cached_data, chunk_factor = self.get_all_data(shard=shard, shuffled=False)
            except FileNotFoundError:
                raise FileNotFoundError(f"consolidation: no data found for {self.split} and shard {shard}")

            logger.info(cached_data.shape)
            all_data.append(cached_data)
            logger.info(batch_begin)
            batch_begin = batch_end
        all_data_array = np.concatenate(tuple(all_data))
        logger.info(all_data_array.shape)
        np.random.shuffle(all_data_array)
        logger.info(f" {self.split} data was consolidated, shuffled and saved!")
        np.save(
            os.path.join(self.PATH_CHUNKED_DATA,
                         f"{self.dataset}_{self.split}_full_chunk_size_{self.chunk_size}_stride_{self.stride}_last_{not self.discard_last}_first_{not self.discard_first}_shuffled.npy"),
            all_data_array)

    def chunk_data_anew(self):
        """
        Chunks raw dataset into tokenized segments for efficient data processing based on
        configuration parameters specific to the dataset type and usage split.

        This function performs processing of the dataset by:
            - Checking the validity and existence of raw dataset files.
            - Loading and filtering the dataset according to the dataset type
              specified (`pg19` or `fineweb`).
            - Tokenizing the input text data using a provided tokenizer.
            - Segmenting tokenized data into manageable chunks for subsequent usage.

        The behavior of the function depends largely on whether it is executed with the
        `training` or `evaluation` split. Based on the split, parameters such as stride,
        data subset size, and data loading limits are configured.

        The function logs its progress and ensures that datasets are properly tokenized
        and chunked, ready for further model training or evaluation processes.

        :param self: Instance containing the dataset properties and tokenizer details.
        """
        if self.split == "training":
            dataset_size = self.no_train
            no_of_books = self.train_load_size
        else:
            raise ValueError(f"only split = training is supported, but split = {self.split}")

        logger.info("starting to chunk data anew")
        if self.dataset == "pg19":
            if os.path.exists(self.PATH_RAW_DATA) and len(os.listdir(self.PATH_RAW_DATA)) > 0:
                origin = self.PATH_RAW_DATA
                logger.info(f"loading local dataset from cache: {origin}")
                dataset = load_dataset(self.dataset, trust_remote_code=True,
                                       split=(self.split if self.split == "validation" else "train"),
                                       num_proc=8, cache_dir=self.PATH_RAW_DATA)
            else:
                logger.info(f"{self.PATH_RAW_DATA} is empty! Downloading from Huggingface")
                os.makedirs(self.PATH_RAW_DATA)
                origin = "deepmind/pg19"
                dataset = load_dataset(origin, trust_remote_code=True,
                                       split=(self.split if self.split == "validation" else "train"),
                                       num_proc=8, cache_dir=self.PATH_RAW_DATA)

        else:
            raise ValueError(f"Dataset name {self.dataset} unknown!")

        if dataset_size is None:
            dataset_size = len(dataset["text"])

        k = 0
        while k < dataset_size:
            next_k = min(k + no_of_books, dataset_size)
            logger.info(f"starting to tokenize documents from {k} to {next_k}")
            input_id_lists = self.tokenizer(dataset["text"][k:next_k], truncation=False)["input_ids"]
            logger.info("tokenization finished")
            self.make_chunks(input_id_lists,
                             detailed_split=f"{self.split}_{k}_{next_k}",
                             )
            k = next_k
            if dataset_size is not None and k >= dataset_size:
                break
        logger.info(f"{self.split} data was freshly chunked!")

    def make_chunks(self, input_id_lists, detailed_split: str) -> np.array:
        """
        Splits the tokenized documents into even-sized chunks. Note that we never put more than one document into the
        same sequence.
        :param input_id_lists: input_ids from the output of a tokenizer
        :param detailed_split: indicates the subset which is currently processed and saved
        :return: an array of size #chunks x chunk_size
        """
        chunk_size = self.chunk_size
        start_idx = 0
        if self.tokenizer.bos_token is not None:
            chunk_size -= 1
            start_idx = 1

        input_ids_chunked = []
        for input_id_list in input_id_lists:
            all_windows = list(
                more_itertools.windowed(input_id_list[start_idx:], chunk_size, fillvalue="!", step=self.stride))
            if self.discard_last:
                if all_windows[-1][-1] == "!":
                    all_windows = all_windows[:-1]
            if self.discard_first:
                all_windows = all_windows[1:]
            input_ids_chunked += all_windows
        as_array = np.array(input_ids_chunked,
                            dtype=np.int32)
        if self.tokenizer.bos_token is not None:
            as_array = np.concatenate(
                (np.ones((len(as_array), 1), dtype=np.int32) * self.tokenizer.bos_token_id, as_array),
                axis=1, dtype=np.int32)

        if self.tokenizer.bos_token is not None:
            chunk_size += 1
        assert chunk_size == self.chunk_size
        save_path = os.path.join(self.PATH_CHUNKED_DATA,
                                 f"{self.dataset}_{detailed_split}_chunk_size_{chunk_size}_stride_{self.stride}_last_{not self.discard_last}_first_{not self.discard_first}.npy")
        if os.path.isfile(save_path):
            logger.info(f"saving {save_path} exists already. Will be overwritten!")
            os.remove(save_path)
        np.save(save_path, as_array)
        return as_array

    def frozen_collect_and_make_hf_dataset(self,
                                           precomputed_weights_path: str = None,
                                           folder_non_frozen: str = None,
                                           no_chunks: int = None, no_devices: int = 8):
        """

        After precomputing the weights, there is a folder with single sequence .npz files. In this function,
        they will be collected, combined and saved as a hf dataset which uses the same order as the non-frozen one.

        :param precomputed_weights_path: where the precomputed weights have been saved
        :param folder_non_frozen: where the original hf dataset is stored from which the frozen loss values were calculated
        :param no_chunks: if None, use all there are. Otherwise, will throw an error if not all files are found
        :param no_devices: the number of GPUs the scoring was performed with
        :return:
        """

        if self.data_subset is not None:
            full_dataset_name = f"{self.dataset}-{self.data_subset}"
        else:
            full_dataset_name = self.dataset

        if precomputed_weights_path is None:
            precomputed_weights_path = f"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/{full_dataset_name}_chunked/llama3/precomputed_weights"
        if folder_non_frozen is None:
            folder_non_frozen = f"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/{full_dataset_name}_chunked/llama3"

        frozen_indicator = False

        datapath_non_frozen = os.path.join(folder_non_frozen,
                                           f"{full_dataset_name}_{self.split}_chunk_size_{self.chunk_size}_stride_{self.stride}_last_{not self.discard_last}_first_{not self.discard_first}_frozen_{frozen_indicator}.hf")
        logger.info(f"Attempting to read from {datapath_non_frozen}")
        original_data = load_from_disk(datapath_non_frozen)

        set_random_state(42)

        all_data = {}

        chunk = 0
        chunks_over = False
        while not chunks_over:
            if no_chunks is not None and chunk >= no_chunks:
                break
            for device in range(no_devices):
                filepath = os.path.join(precomputed_weights_path, f"precomputed_weights_{chunk}_{device}.npz")
                if os.path.isfile(filepath):
                    saved_array = np.load(filepath)
                    key = saved_array["input_ids"].tobytes()
                    all_data[key] = {"cross_entropy": saved_array["cross_entropy"]}

                else:
                    if no_chunks is None:
                        chunks_over = True
                    else:
                        raise Exception(f"precomputed_weights_{chunk}_{device} does not exist")
            chunk += 1
            print(f"done with chunk {chunk}")


        def frozen_gen():
            for datapoint in original_data:
                datapoint = np.array(datapoint["input_ids"])
                key = datapoint.tobytes()
                yield {"input_ids": datapoint,
                       "cross_entropy": all_data[key]["cross_entropy"]}

        train_data = Dataset.from_generator(frozen_gen,
                                            cache_dir="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/caches/dataset_creation")
        print("after dataset generation")
        frozen_indicator = self.scoring_model
        datapath = os.path.join(folder_non_frozen,
                                f"{full_dataset_name}_{self.split}_chunk_size_{self.chunk_size}_stride_{self.stride}_last_{not self.discard_last}_first_{not self.discard_first}_frozen_{frozen_indicator}.hf")
        train_data.save_to_disk(datapath)
