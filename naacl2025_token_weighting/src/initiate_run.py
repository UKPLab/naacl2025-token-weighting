#!/usr/bin/env python3
import argparse
import json
import os

from training import train
from run_settings import get_settings
from logger_setup import initialize_logger
from data_preprocessing import precompute_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with given parameters.")

    parser.add_argument("--work_dir",      type=str, required=False,
                        help="Path to the working directory.")
    parser.add_argument("--resume",        type=str, required=False, default=False,
                        help="resume from a checkpoint?")
    parser.add_argument("--checkpoint_no", type=int, required=False, default=None,
                        help="Checkpoint number to resume from.")

    # for deepspeed compatibility. are not used in the following
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()

    if not args.work_dir:
        if "WORKDIR" in os.environ:
            workdir = os.environ["WORKDIR"]
        else:
            raise ValueError(
                "The working directory must be specified via --work_dir or the WORKDIR environment variable.")
    else:
        workdir = args.work_dir

    mode = "_".join(workdir.split("/")[-1].split("_")[3:])
    time = "_".join(workdir.split("/")[-1].split("_")[1:3])

    if args.resume == "True" or args.resume == True:
        resume = {"checkpoint": args.checkpoint_no}
    elif args.resume == "False" or args.resume == False:
        resume = None
    else:
        raise ValueError(f"Unknown value for resume: {args.resume}")

    settings = get_settings(mode, resume, workdir, time)

    initialize_logger()

    if settings.precompute_weights:
        precompute_weights(settings)
    else:
        train(settings)
