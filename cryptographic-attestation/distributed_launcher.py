#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import uuid
from argparse import ArgumentParser, REMAINDER

import torch


"""
Wrapper to launch MPC scripts as multiple processes.
"""


def main():
    args = parse_args()

    gpus = [int(s) for s in args.gpus.split(",")]
    assert len(gpus) == args.world_size + 1, "Number of GPUs must match number of parties"

    for gpu in gpus:
        assert gpu >= 0, "GPU IDs must be non-negative"
        assert gpu < torch.cuda.device_count(), "GPU ID must be less than number of GPUs"

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["WORLD_SIZE"] = str(args.world_size)

    processes = []

    # Use random file so multiple jobs can be run simultaneously
    INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())

    for rank in range(0, args.world_size):  # computation process
        # each process's rank
        current_env["RANK"] = str(rank)
        current_env["RENDEZVOUS"] = INIT_METHOD

        # spawn the processes
        cmd = ["python", args.training_script] + args.training_script_args + [f"--gpu={gpus[rank]}"]

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    # TTP process
    current_env["RANK"] = str(args.world_size)
    current_env["RENDEZVOUS"] = INIT_METHOD
    cmd = ["python", "launch_TTPServer.py", f"--gpu={gpus[args.world_size]}"]
    process = subprocess.Popen(cmd, env=current_env)
    processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=process.args
            )


def parse_args():
    parser = ArgumentParser()

    # Optional arguments for the launch helper
    parser.add_argument("--world_size", type=int, default=2, help="The number of computation parties")
    parser.add_argument("--gpus", type=str, default="", help="Index of GPUs to use (comma-separated)")

    # positional
    parser.add_argument("training_script", type=str, help="The full path to the single GPU training program/script to be launched in parallel, followed by all the arguments for the training script")

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    main()
