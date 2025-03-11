import os
from data_preprocessing import DataPreprocessor
from logger_setup import initialize_logger
from run_settings import get_settings

initialize_logger()

tokenizer_name = "llama3"
no_devices = 1 # the number of GPUs that were used to score the data

PATH_RAW_DATA = "../data"
PATH_CHUNKED_DATA = os.path.join(PATH_RAW_DATA, f"preprocessed_{tokenizer_name}")
PATH_FROZEN_CACHE = os.path.join(PATH_CHUNKED_DATA, f"precomputed_weights_3.0_8B")

settings = get_settings(mode="llama3_32k_dense_precompute_weights")
data_preprocessor = DataPreprocessor(settings=settings)

data_preprocessor.frozen_collect_and_make_hf_dataset(precomputed_weights_path = PATH_FROZEN_CACHE,
                                           folder_non_frozen = PATH_CHUNKED_DATA,
                                           no_devices = no_devices)