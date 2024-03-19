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

import matplotlib.pyplot as plt
import argparse
import logging
import os
import numpy as np
import torch
from pathlib import Path
from typing import List
import scipy.stats
import random
import csv
import pandas as pd
from tqdm import tqdm
from . import meta_utils
from . import os_layer
from . import models
from . import data
from . import attestation_dataset_utils


def gather_metric(args, csv_filename):
    df = pd.read_csv(csv_filename)
    print(df)
    return df[args.metric].tolist()


def save_csv(args, results_per_window, filter_val=''):
    # Look at each window for this particular filter
    if args.dataset == 'ARXIV':
        properties = ["Window Size", "9", "10", "11", "12", "13", "14", "15", "16", "17"]
    elif args.dataset == 'BONEAGE':
        properties = ["Window Size", "02", "03","04", "05", "06", "07", "08"]
    else:
        properties = ["Window Size", "00", "01", "02", "03","04", "05", "06", "07", "08", "09", "10"]

    # Put in the window sizes as the first column
    for i, (window_size, li) in enumerate(zip(args.window_sizes, results_per_window)):
        results_per_window[i].insert(0,int(window_size))
    
    file_name = os.path.join(Path(args.plots_path), args.dataset+'_'+args.metric+'_'+filter_val+'_comparison')
    with open(file_name+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(properties)
        writer.writerows(results_per_window)
    print(f'Saved to: {file_name+".csv"}')

    # for win_size, res in zip(results_per_window, args.window_sizes):
        # print(res)

def generate_final_table(args, all_csvs):
    # all_csvs is a list of csvs. The number of csvs in each of the lists should be the same
    # Each list in all_csvs refers to one of the window sizes

    gathered_values_per_filter = []
    for per_window in zip(*all_csvs):
        # The same filter (eg. race) for each window size (item can would have race for each window size being compared)
        gathered_values_per_window = []
        for window in per_window:
            print(window)
            result_for_window = gather_metric(args, window)
            gathered_values_per_window.append(result_for_window)
        gathered_values_per_filter.append(gathered_values_per_window)
    
    print(gathered_values_per_filter) # num_filters x num_windows x num_ratios

    if len(gathered_values_per_filter) == 1:
        return save_csv(args, gathered_values_per_filter[-1])

    print(f'len gathered_values_per_filter: {len(gathered_values_per_filter)}')
    for i, filter_results in enumerate(gathered_values_per_filter):
        print(i)
        filter_val = 's'
        if i == 1:
            filter_val = 'r'
        print(f'Saving with filter_val {filter_val}')
        save_csv(args, filter_results, filter_val)



def assemble_window_size_table(args, log):
    all_csvs = []
    for suffix in args.dataset_suffixes:
        # Census is going to have two csvs (one for race, the other for sex)
        current_csvs = []
        dataset_dir_name = args.dataset + suffix
        plots_path: Path = Path(args.plots_path) / dataset_dir_name
        paths_in_folder = os.listdir(plots_path)

        for path in paths_in_folder:
            if path.endswith('.csv'):
                current_csvs.append(os.path.join(plots_path, path))
        all_csvs.append(current_csvs)

    generate_final_table(args, all_csvs)



def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--plots_path', type=str, default='plots/', help="Directory to store the plots")
    parser.add_argument('--dataset', type=str, default="ARXIV", help='Options: CENSUS, BONEAGE, ARXIV')
    parser.add_argument('--dataset_suffixes', nargs='+', default=[''], help='CENSUS, BONEAGE, or ARXIV could have some suffix at the end for their directory. We want a list of these')                     
    parser.add_argument('--window_sizes', nargs='+', default=[''], help='The window sizes corresponding to each of the suffixes')
    # parser.add_argument('--claim_property', default="12")
    parser.add_argument('--metric', type=str, default='Acceptance AUCROC', help='Metric that we are comparing between the windows')
    parser.add_argument('--filter', type=str,required=False,default='race',help='which filter to use')
    # parser.add_argument('--logfile_suffix', default="test")
    args: argparse.Namespace = parser.parse_args()
    assert(len(args.dataset_suffixes)==len(args.window_sizes))
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="window_size_tables.log", filemode="w")
    log: logging.Logger = logging.getLogger("PropertyAttestation")
    args = handle_args()
    assemble_window_size_table(args,log)