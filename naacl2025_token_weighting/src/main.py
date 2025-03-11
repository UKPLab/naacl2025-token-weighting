import argparse
import os
import json
import torch

from datetime import datetime

from utils import create_subprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for running training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="../runs",
        help="Path to the output directory where runs will be saved."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="llama3_32k_dense",
        help="Name of the json file with run configs"
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="deepspeed",
        help="Name of the launcher. Supported values are 'deepspeed' and 'python' and 'accelerate launch'"
    )
    parser.add_argument(
        "--resumed_run_path",
        type=str,
        default="",
        help="Existing run path if training is resumed from checkpoint. Only meaningful if resume_from_checkpoint is True"
    )
    parser.add_argument(
        "--checkpoint_number",
        type=int,
        default=0,
        help="Existing checkpoint number where training is resumed from. Only meaningful if resume_from_checkpoint is True"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from the provided checkpoint if set."
    )

    args = parser.parse_args()

    OUT_PATH = os.path.abspath(args.out_path) if not os.path.isabs(args.out_path) else args.out_path
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    if args.resume_from_checkpoint is None or args.resume_from_checkpoint is False:
        resume_from_checkpoint = False
    else:
        resume_from_checkpoint = True

    resumed_run_path = os.path.abspath(args.resumed_run_path) if args.resumed_run_path and not os.path.isabs(
        args.resumed_run_path) else args.resumed_run_path
    checkpoint_no = int(args.checkpoint_number)
    execute_as = args.launcher

    run_name = args.run_name

    if resume_from_checkpoint:
        WORK_DIR = resumed_run_path
        print(f"Resuming run {resumed_run_path} from checkpoint {checkpoint_no}")
    else:
        timestamp = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        WORK_DIR = os.path.join(OUT_PATH, f"run_{timestamp}_{run_name}")
        print(f"Starting new training run based on {run_name}.json. Will be saved in {WORK_DIR}")
        os.makedirs(WORK_DIR)



    bash_command = f"{execute_as} initiate_run.py --work_dir {WORK_DIR} --resume {resume_from_checkpoint} --checkpoint_no {checkpoint_no}"
    print(bash_command)
    exit_code = create_subprocess(bash_command)

    if exit_code == 0 and json.load(open(f"../configs/{run_name}.json"))["training_dynamics"]["precompute_weights"]:
        create_subprocess(f"python add_frozen_base_to_dataset.py --no_devices {torch.cuda.device_count()}")
