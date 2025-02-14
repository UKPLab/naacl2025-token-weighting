import os
import random
import numpy as np
import torch

from subprocess import Popen, PIPE
from typing import Dict
from transformers import EvalPrediction, enable_full_determinism, set_seed, LlamaConfig, PhiConfig, AutoModelForCausalLM

from trainer import ShortLongLoss

def get_model(settings, model_path:str = None):
    if model_path is None:
        model_path = settings.model_path
    if model_path in ["microsoft/phi-2"]:
        config = PhiConfig.from_pretrained(model_path, rope_theta=settings.rope_theta,
                                           max_position_embeddings=settings.chunk_size if not settings.precompute_weights else settings.custom_loss.base_length,
                                           rope_scaling=None,
                                           token=settings.access_token,
                                           )
    elif model_path in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.2-1B",
                                 "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"]:
        config = LlamaConfig.from_pretrained(model_path, rope_theta=settings.rope_theta,
                                             max_position_embeddings=settings.chunk_size if not settings.precompute_weights else settings.custom_loss.base_length,
                                             rope_scaling=None,
                                             token=settings.access_token,
                                             )
    else:
        raise ValueError(f"Unknown model_path {model_path}")

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map=None,
                                                 low_cpu_mem_usage=True if settings.deepspeed_stage is None else False,
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=settings.precision,
                                                 config=config,
                                                 token=settings.access_token
                                                 )

    model.config.pad_token_id = settings.tokenizer.pad_token_id

    return model


def get_loss(all_settings: Dict) -> ShortLongLoss:
    if all_settings["loss"]["name"] is None:
        loss = None
    else:
        loss = ShortLongLoss(**all_settings["loss"]["params"])
    return loss


def logloss(eval_pred: EvalPrediction):
    metric = torch.nn.CrossEntropyLoss(reduction="mean")
    preds = eval_pred["predictions"][:, :-1, :]
    input_ids = eval_pred["inputs"][:, 1:]
    error = metric(torch.transpose(torch.tensor(preds), 1, 2),
                   torch.tensor(input_ids, dtype=torch.long)).detach().cpu().numpy()
    return {"avg_logloss": error}


def set_random_state(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    enable_full_determinism(seed, warn_only=False)


def create_subprocess(subprocess_args):
    with Popen(subprocess_args, stdout=PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            print(line, end='')  # process line here
