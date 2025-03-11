import os
import json
import torch

from transformers import AutoTokenizer
from datetime import datetime
from huggingface_hub import login

from trainer import ShortLongLoss
from utils import get_loss


class RunSettings:
    def __init__(self,
                 log: str = "aim",
                 logging_project_name: str = "",
                 run_name: str = "",
                 save_models: bool = True,
                 save_every: int = 30,
                 model_checkpoint: str = "",
                 use_model_checkpoint: bool = False,
                 use_frozen_base: bool = False,
                 custom_loss: ShortLongLoss = None,
                 discard_last: bool = True,
                 discard_first: bool = False,
                 max_chunk_factor: int = 16,
                 no_train: int = None,
                 num_train_sequences: int = 1000,
                 chunk_size: int = 8192,
                 stride: int = 6144,
                 no_update_steps: int = 250,
                 grad_clip: float = 1.0,
                 warmup_steps: int = 20,
                 min_lr_ratio: float = 0.0,
                 adam_beta1=0.9,
                 adam_beta2=0.95,
                 adam_epsilon=1e-8,
                 weight_decay=0.0,
                 minimum_decay_factor: float = 0.1,
                 gradient_checkpointing: bool = False,
                 deepspeed_stage: int = 0,
                 deepspeed_config_file: str = "deepspeed_config.json",
                 make_decay: bool = True,
                 precision_name="fp16",
                 scheduler: bool = True,
                 do_train: bool = True,
                 rope_theta:float=500000.0,
                 lr: float = None,
                 model_path: str = None,
                 batch_size: int = None,
                 grad_acc: int = None,
                 dataset: str = "pg19",
                 scoring_model: str = "",
                 PATH_CHUNKED_DATA: str = None,
                 WORK_DIR: str = None,
                 resume_from_checkpoint=False,
                 non_print_logger=True,
                 lr_scheduler_type="constant",
                 lr_scheduler_kwargs=None,
                 precompute_weights=False,
                 preprocess_data=False,
                 seed=42,
                 scoring_minibatch=1,
                 time= None,
                 tokenizer_name=None,
                 tokenizer = None
                 ):

        # hyperparams logging
        self.seed = seed
        self.time = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}' if time is None else time
        self.logging_project_name = logging_project_name

        assert log in ["aim", None], f"Unknown value for external logger: {log}"
        self.log = log
        self.run_name = run_name
        self.save_models = save_models
        self.save_every = save_every

        self.model_checkpoint = model_checkpoint
        self.use_model_checkpoint = use_model_checkpoint

        self.access_token = os.getenv("HF_TOKEN")
        if os.getenv("HF_HUB_OFFLINE") != "1":
            login(token=self.access_token)
        self.non_print_logger = non_print_logger

        self.model_path = model_path
        self.use_frozen_base = use_frozen_base
        self.rope_theta = rope_theta

        str_to_precision_dict = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self.precision_name = precision_name
        self.precision = str_to_precision_dict[self.precision_name]
        self.grad_acc = grad_acc
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.minimum_decay_factor = minimum_decay_factor
        self.scheduler = scheduler
        self.make_decay = make_decay
        self.gradient_checkpointing = gradient_checkpointing
        self.batch_size = batch_size
        self.lr = lr

        self.deepspeed_stage = deepspeed_stage
        self.deepspeed_config_file = deepspeed_config_file
        self.resume_from_checkpoint = resume_from_checkpoint
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.precompute_weights = precompute_weights
        self.preprocess_data = preprocess_data
        self.scoring_minibatch = scoring_minibatch
        self.custom_loss = custom_loss

        # hyperparameters data
        self.discard_last = discard_last
        self.discard_first = discard_first
        self.max_chunk_factor = max_chunk_factor
        self.scoring_model = scoring_model

        # hyperparameters train data
        self.dataset = dataset
        self.no_train = no_train
        self.num_train_sequences = num_train_sequences
        self.chunk_size = chunk_size
        self.stride = stride

        # hyperparameters eval data
        self.do_train = do_train

        if tokenizer_name is not None and tokenizer is not None:
            self.tokenizer_name = tokenizer_name
            self.tokenizer = tokenizer

        elif "llama-3" in self.model_path.lower():
            self.tokenizer_name = "llama3"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left", use_fast=True, token = self.access_token)
        elif "phi" in self.model_path.lower():
            self.tokenizer_name = "phi2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left", use_fast=True,
                                                           token=self.access_token, is_split_into_words=False,
                                                           add_prefix_space=False)
        # training
        self.no_update_steps = no_update_steps

        # data handling
        self.PATH_CHUNKED_DATA = f"../data/preprocessed_{self.tokenizer_name}" if PATH_CHUNKED_DATA is None else PATH_CHUNKED_DATA
        self.WORK_DIR = WORK_DIR


def get_settings(mode: str, resume: dict = None, workdir: str = None, time=None) -> RunSettings:
    if resume is None:
        all_settings = json.load(open(f"../configs/{mode}.json"))
        loss = get_loss(all_settings)
        settings = RunSettings(
            run_name=mode,
            custom_loss=loss,
            WORK_DIR=workdir,
            time=time,
            **{**all_settings["logging"],
               **all_settings["model"],
               **all_settings["data_preprocessing"],
               **all_settings["training_dynamics"],
               **all_settings["deepspeed"]
               }
        )

        all_settings["paths"]["WORK_DIR"] = settings.WORK_DIR
        if workdir is not None:
            json.dump(all_settings, open(os.path.join(settings.WORK_DIR, "settings.json"), "w"), indent=4)
    else:
        checkpoint_no = resume["checkpoint"]
        all_settings = json.load(open(os.path.join(workdir, "settings.json")))
        loss = get_loss(all_settings)
        settings = RunSettings(
            run_name=mode,
            custom_loss=loss,
            **{**all_settings["logging"],
               **all_settings["model"],
               **all_settings["data_preprocessing"],
               **all_settings["training_dynamics"],
               **all_settings["deepspeed"]},
            resume_from_checkpoint=os.path.join(workdir, f"checkpoint-{checkpoint_no}"))
    return settings
