import os
import json
import numpy as np
import torch

from aim.hugging_face import AimCallback
from transformers import AutoModelForCausalLM, set_seed, TrainingArguments, default_data_collator, LlamaConfig, \
    TrainerCallback, PhiConfig, AutoTokenizer
from trainer import CustomTrainer
from run_settings import RunSettings
from utils import logloss, set_random_state,  get_model
from data_preprocessing import get_data
from logger_setup import logger


def train(settings: RunSettings):
    set_random_state(settings.seed)

    if settings.deepspeed_stage >= 0:
        deepspeed_config_file = json.load(open(f"../configs/{settings.deepspeed_config_file}"))
        deepspeed_config_file["zero_optimization"]["stage"] = settings.deepspeed_stage
    else:
        deepspeed_config_file = None

    args = TrainingArguments(
        output_dir=settings.WORK_DIR,
        do_train=settings.do_train,
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_acc,
        learning_rate=settings.lr,
        adam_beta1=settings.adam_beta1,
        adam_beta2=settings.adam_beta2,
        adam_epsilon=settings.adam_epsilon,
        weight_decay=settings.weight_decay,
        max_grad_norm=settings.grad_clip,
        lr_scheduler_type=settings.lr_scheduler_type,
        lr_scheduler_kwargs=settings.lr_scheduler_kwargs,
        warmup_steps=settings.warmup_steps,
        log_level="info",
        bf16=settings.precision == torch.bfloat16,
        fp16=settings.precision == torch.float16,
        report_to="none",
        gradient_checkpointing=settings.gradient_checkpointing,
        deepspeed=deepspeed_config_file,
        eval_accumulation_steps=None,
        run_name=settings.run_name,
        save_strategy="steps" if settings.save_models else "no",
        save_steps=settings.save_every if settings.save_models else None,
        logging_strategy="steps" if settings.log is not None else "no",
        logging_steps=1,
        max_steps=settings.no_update_steps,
        prediction_loss_only=True,
        remove_unused_columns=not settings.use_frozen_base,
        seed=settings.seed,
        local_rank=int(os.getenv("LOCAL_RANK", -1)),
    )

    model = get_model(settings)

    train_data, _ = get_data(settings)

    aim_callback = AimCallback(experiment=settings.logging_project_name)

    trainer = CustomTrainer(
        model=model,
        settings=settings,
        args=args,
        train_dataset=train_data,
        compute_metrics=logloss,
        callbacks=[aim_callback]
    )

    if trainer.is_world_process_zero():
        aim_callback.experiment["json_name"] = settings.run_name
        aim_callback.experiment["filepath"] = settings.WORK_DIR

    if settings.do_train:
        trainer.train(resume_from_checkpoint=settings.resume_from_checkpoint)

    logger.info(f"Training finished successfully")
