# Authors: Vasisht Duddu, Anudeep Das, Nora Khayata, Hossein Yalame, Thomas Schneider, N Asokan
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import statistics
import argparse

def find_matching_log_files(directory, pattern):
    """
    Find all of the log files in a directory whose names all fit a certain regex pattern
    """
    matching_files = []
    for file in os.listdir(directory):
        if file.startswith(pattern) and file.endswith('.log'):
            matching_files.append(os.path.join(directory, file))
    return matching_files


def extract_running_times(args, log_file, get_time = True):
    running_times_dist = []
    running_times_training = []
    dist_comm = []
    train_comm = []
    with open(log_file, 'r') as file:
        lines = file.readlines()
        # Read in reverse: get the last 5 lines

        if get_time:
            for i in range(len(lines) - 1):
                if lines[i].strip() == "Distributional Property Check Results":
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("Run Time:"):
                        running_time = next_line.split(":")[1].strip()
                        running_times_dist.append(float(running_time))
                    second_line = lines[i+2].strip()
                    print('Distribution Check Communication Stats coming Next')
                    print(second_line)
                elif args.dataset == 'bone' and lines[i].strip() == "Training":
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("Run Time:"):
                        running_time = next_line.split(":")[1].strip()
                        running_times_training.append(float(running_time))
                elif args.dataset == 'census' and lines[i].strip() == "Training Results":
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("Run Time:"):
                        running_time = next_line.split(":")[1].strip()
                        running_times_training.append(float(running_time))
                    # second_line = lines[i+2].strip()
                    # print('Training Communication Stats coming Next')
                    # print(second_line)


    return running_times_dist, running_times_training


def calculate_mean_and_stdev(arr, type):
    print(arr)

    # Calculate mean
    mean = statistics.mean(arr)

    # Calculate standard deviation
    stdev = statistics.stdev(arr)

    # Print the mean and standard deviation
    print(f"Mean of {type} over {len(arr)} values:", mean)
    print(f"Standard Deviation of {type} over {len(arr)} values:", stdev)

    return mean, stdev


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--logfiles_dir', type=str, default='.', help="Directory holding all of the log files")
    # parser.add_argument('--logfile_suffix', default="", help = 'The logfile may have some text before .log, that text is represented in')
    parser.add_argument('--dataset', type=str, default="bone", help='Options: bone, census')
    parser.add_argument('--time_check', action='store_true', default=False, help='Get the distribution check times')
    # parser.add_argument('--claim_property', default="122")
    parser.add_argument('--filter', type=str,default="",required=False,help='which filter to use. Options: r, s')
    args: argparse.Namespace = parser.parse_args()
    return args
    

if __name__ =='__main__':
    args = handle_args()

    if args.filter == "":
        pattern = f'{args.dataset}_run'
    else:
        pattern = f'{args.dataset}_{args.filter}_run' 
    print(pattern)

    log_files = find_matching_log_files(args.logfiles_dir, pattern)
    dist_check_times = []
    training_times = []
    # train_comms = []
    # dist_comms = []
    print(f'Looking through {len(log_files)} files')
    for logfile in log_files:
        dist_check_time, training_time  = extract_running_times(args, logfile, args.time_check)
        if len(dist_check_time) > 0:
            dist_check_times += dist_check_time
        if len(training_time) > 0:
            training_times += training_time
        # if len(dist_comm) > 0:
        #     dist_comms.append(dist_comm)
        # if len(train_comm) > 0:
        #     train_comms.append(train_comm)

    if len(dist_check_times) > 0:
        calculate_mean_and_stdev(dist_check_times, "Distribution Check")
    print()
    if len(training_times) > 0:
        calculate_mean_and_stdev(training_times, "Training Time")
    # print()
    # if len(dist_comms) > 0:
    #     calculate_mean_and_stdev(dist_comms, "Distribution Check Communication (Bytes)")
    # print()
    # if len(train_comms) > 0:
    #     calculate_mean_and_stdev(train_comms, "Training Communication (Bytes)")


    

    

