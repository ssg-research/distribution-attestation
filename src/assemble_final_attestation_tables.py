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
from pathlib import Path
import os
import csv
import numpy as np
import pandas as pd
from . import parse_logs

def get_ft(file_path):
    to_ret = [-1,-1,-1,-1] # fpr, tpr, fnr, tnr
    with open(os.path.join(file_path, 'ft_rates.npy'), 'rb') as f:
        to_ret[0] = np.load(f)
        to_ret[1] = np.load(f)
        to_ret[2] = np.load(f)
        to_ret[3] = np.load(f)
    return to_ret

def get_tar_trr_numbers(ft_rates, returned):
    fpr, tpr, fnr, tnr = ft_rates
    tpr_at_fpr001 = tpr[np.where(fpr.round(2)<=0.01)]
    tpr_at_fpr005 = tpr[np.where((fpr.round(2)<=0.05))]
    tpr_at_fpr01 = tpr[np.where((fpr.round(2)<=0.1))]
    print(tpr_at_fpr001)
    val_tpr_at_fpr001 = tpr_at_fpr001.tolist()[-1]
    val_tpr_at_fpr005 = tpr_at_fpr005.tolist()[-1]
    val_tpr_at_fpr01 = tpr_at_fpr01.tolist()[-1]
    print(f'TAR at 1%: {val_tpr_at_fpr001}')
    print(f'TAR at 5%: {val_tpr_at_fpr005}')
    print(f'TAR at 10%: {val_tpr_at_fpr01}') 

    tnr_at_fnr001 = tnr[np.where(fnr.round(2)<=0.01)]
    tnr_at_fnr005 = tnr[np.where((fnr.round(2)<=0.05))]
    tnr_at_fnr01 = tnr[np.where((fnr.round(2)<=0.1))]
    val_tnr_at_fnr001 = tnr_at_fnr001.tolist()[-1]
    val_tnr_at_fnr005 = tnr_at_fnr005.tolist()[-1]
    val_tnr_at_fnr01 = tnr_at_fnr01.tolist()[-1]
    print(f'TRR at FRR 1%: {val_tnr_at_fnr001}')
    print(f'TRR at FRR 5%: {val_tnr_at_fnr005}')
    print(f'TRR at FRR 10%: {val_tnr_at_fnr01}')
    returned.append(round(val_tpr_at_fpr001,2))
    returned.append(round(val_tpr_at_fpr005,2))
    returned.append(round(val_tpr_at_fpr01,2))
    returned.append(round(val_tnr_at_fnr001,2))
    returned.append(round(val_tnr_at_fnr005,2))
    returned.append(round(val_tnr_at_fnr01,2))
    print(f'returned: {returned}')
    return returned




def assemble_tables(args, log):
    if args.dataset == 'ARXIV':
        properties = ["9", "10", "11", "12", "13", "14", "15", "16", "17"]
    elif args.dataset == 'BONEAGE':
        properties = ["02", "03","04", "05", "06", "07", "08"]
    else:
        properties = ["00", "01", "02", "03","04", "05", "06", "07", "08", "09", "10"]
    
    dataset_str = args.dataset + args.dataset_suffix
    
    plots_path: Path = Path(args.plots_path) / dataset_str / "Property Attestation"
    ret_vals = []
    frr_array = []
    trr_array = []
    for p in properties:
        args.claim_property = p
        if not args.do_not_make_tables:
            returned = [i[0] for i in list(parse_logs.parse_and_collect(args,log)) if len(i) > 0]
            returned.insert(0, p)
        else:
            returned = []

        
        if args.dataset == 'BONEAGE':
            ft_rates = get_ft(os.path.join(plots_path, str(int(p)/10)))
        elif args.dataset == 'CENSUS':
            ft_rates = get_ft(os.path.join(plots_path, args.filter, str(int(p)/10)))
        else:
            ft_rates = get_ft(os.path.join(plots_path, p))
        
        frr_array.append(ft_rates[-2])
        trr_array.append(ft_rates[-1])
        if args.make_plots:
            # plot(args, ft_rates[0], ft_rates[1], plots_path)
            plot(args, frr_array, trr_array, properties, plots_path, )
        returned = get_tar_trr_numbers(ft_rates, returned)



        ret_vals.append(returned)

    if not args.do_not_make_tables:
        parse_logs.save_tables(args, ret_vals, add_ratio=True, at_low_ratio = True)
    else:
        return ret_vals
    
    
def plot(args, fr, tr, properties, plots_path='.', color='black', ls='solid', label='', prefix='acceptance', suffix=''):
    fig = plt.figure(figsize=(4.5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'y']
    linestyle = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashed', 'dashed']
    ax.plot([0,1], [0,1], linestyle='--', color='black', label='Random')
    for p, frval, trval, col, ls in zip(properties, fr, tr, colors, linestyle):
        from sklearn import metrics
        aucscore = metrics.auc(frval, trval)
        ax.plot(frval, trval, ls=ls, marker='.', markersize=2, label=p + ': {:.2f}'.format(aucscore), c=col)
    # ax.axvline(x=0.05, ymin=0.0, ymax=1.0, linewidth=2, color='black', linestyle='solid', label='TRR@5% FRR')
    # ax.axhline(y=0.95, xmin=0.0, xmax=1.0, linewidth=2, color='black', linestyle='dotted', label='TAR@5% FAR')
    ax.legend(loc='best',fontsize="8")
    ax.set_xlabel("False Rejection Rate", fontsize=14)
    ax.set_ylabel("True Rejection Rate",fontsize=14)
    save_path = os.path.join('.',"{}_{}_attestation_aucroc.pdf".format(args.dataset, args.filter))
    plt.savefig(save_path, bbox_inches = 'tight',pad_inches = 0, dpi=400)
    print(f"Saved to {save_path}")


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--plots_path', type=str, default='plots_new/', help="Directory to store the plots")
    parser.add_argument('--dataset', type=str, default="ARXIV", help='Options: CENSUS, BONEAGE, ARXIV')
    parser.add_argument('--dataset_suffix', type=str, default='_win1', help='CENSUS, BONEAGE, or ARXIV could have some suffix at the end for their plots directory')                     
    # parser.add_argument('--claim_property', default="12")
    parser.add_argument('--filter', type=str,required=False,default='',help='which filter to use')
    parser.add_argument('--logfile_suffix', default="test")
    parser.add_argument('--logfile_subdir', default='', type=str, help='Sometimes the logifle may be in a subdirectory within src')
    parser.add_argument('--do_not_make_tables', default=False, action='store_true', help='This is for when only ft_rates is to be parsed')
    parser.add_argument('--make_plots', default=False, action='store_true', help='Make plots out of the ft_rates')
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="train_verifier_model_exp.log", filemode="w")
    log: logging.Logger = logging.getLogger("PropertyAttestation")
    args = handle_args()
    assemble_tables(args,log)