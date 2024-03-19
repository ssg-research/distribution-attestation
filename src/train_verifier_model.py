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
import torch.nn as nn
from pathlib import Path
from typing import List
import scipy.stats
from sklearn.utils import shuffle
import random
from . import meta_utils
from . import os_layer
from . import models
from . import data
from . import attestation_dataset_utils
from . import robustness
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from timeit import Timer
import gc

import warnings
warnings.filterwarnings("ignore")

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rcParams.update({'mathtext.default':'regular'})

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def train_attestation_model(args, X_train, Y_train, X_test, Y_test, dims):
    best_acc = 0
    best_representation_generator = None
    num_epochs_to_use = args.epochs
    combined_to_use = True
    lr_to_use = 0.001
    if args.dataset == 'BONEAGE':
        num_epochs_to_use = args.epochs_boneage
    elif args.dataset == 'ARXIV':
        num_epochs_to_use = 500
        lr_to_use = 1e-4
    for i in range(args.ntimes):
        representation_generator = models.GenerateRepresentation2(dims).to(device)
        representation_generator, test_acc = meta_utils.train_meta_model(args,representation_generator,(X_train, Y_train),(X_test, Y_test),epochs=num_epochs_to_use, lr=lr_to_use, combined=combined_to_use,batch_size=args.batch_size,eval_every=10)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_representation_generator = representation_generator

    print("Best Representation Generator Accuracy: {}".format(best_acc))
    return best_representation_generator, best_acc



def train_ensemble_model(args, X_train_ens, Y_train, X_test_ens, Y_test, dims, batch_size=1000):
    clf = models.get_model(max_iter=2000)
    clf.fit(X_train_ens, Y_train[:1000])
    score = clf.score(X_test_ens, Y_test[:1000])
    print(f'pred: {clf.predict(X_test_ens)}')
    print(f'Y_test: {Y_test}')
    print(f'Ensemble score with MLPClassifier: {score}')
    return clf, score


def main(args: argparse.Namespace, log: logging.Logger) -> None:

    result_path: Path = Path(args.result_path)
    classifier_save_path: Path = Path(args.classifier_save_path)
    adv_samples_path: Path = Path(args.adv_samples_path)
    if args.dataset == "CENSUS":
        plots_path: Path = Path(args.plots_path) / (args.dataset+args.classifier_save_suffix) / "Property Attestation" / args.filter / args.claim_ratio
        classifier_save_path = classifier_save_path / (args.dataset+args.classifier_save_suffix) / args.filter / args.claim_ratio
    else:
        plots_path: Path = Path(args.plots_path) / (args.dataset+args.classifier_save_suffix) / "Property Attestation" / args.claim_ratio
        if args.dataset == 'ARXIV':
            plots_path: Path = Path(args.plots_path) / (args.dataset+args.classifier_save_suffix) / "Property Attestation" / args.claim_degree
            classifier_save_path = classifier_save_path / (args.dataset+args.classifier_save_suffix) / args.claim_degree
        else:
            classifier_save_path = classifier_save_path / (args.dataset+args.classifier_save_suffix) / args.claim_ratio
    result_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(result_path, log)
    plots_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(plots_path, log)
    classifier_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(classifier_save_path, log)
    if (result_path_status or plots_path_status or classifier_path_status) is None:
        msg: str = f"Something went wrong when creating {result_path} or {plots_path} or {classifier_save_path}. Aborting..."
        log.error(msg)
        raise EnvironmentError(msg)

    if args.dataset == "CENSUS":
        result_path = result_path / "CENSUS"

        targets = ["0.0", "0.1", "0.2", "0.3","0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
        targets = sorted(targets)

        if args.window_alignment == 'centre':
            if float(args.claim_ratio) - args.window_size*0.1 <= 0.0 or float(args.claim_ratio) + args.window_size*0.1 >= 1.0:
                args.use_single_attest = True
        elif args.window_alignment == 'right':
            if float(args.claim_ratio) + args.window_size*0.1 >= 1.0:
                args.use_single_attest = True
        elif args.window_alignment == 'left':
            if float(args.claim_ratio) - args.window_size*0.1 <= 0.0:
                args.use_single_attest = True

        if args.use_single_attest:

            if args.claim_ratio == "0.0" or args.claim_ratio == "1.0":
                args.window_size = 0
            # elif float(args.claim_ratio) >= 0.3 and float(args.claim_ratio) <= 0.7:
            #     args.window_size = 2
                # args.window_alignment = 'right'
            new_targets, targets = attestation_dataset_utils.create_targets(targets, args, window_size=args.window_size, align=args.window_alignment)

            X_train, Y_train, X_test, Y_test, dims, _ = attestation_dataset_utils.assemble_attestation_train_val_data(args, targets, new_targets, result_path, more_adj_samples=args.use_more_adj_samples)

            if args.do_defense:
                X_train, Y_train, X_test, Y_test = robustness.prepare_defense_data(args, adv_samples_path, X_train, Y_train, X_test, Y_test)
            
            best_representation_generator, best_acc = train_attestation_model(args, X_train, Y_train, X_test, Y_test, dims)

            params_verify, labels_verify, label_vals_verify, _ = attestation_dataset_utils.assemble_attestation_test_data(args, targets, new_targets, result_path, return_target_ratio_vals=True)
            y_pred_test, y_test, y_pred_verify, y_verify, thresh , ft_rates, minmax_vals = meta_utils.test_meta_threshold(args, plots_path, best_representation_generator, X_test, Y_test, params_verify, labels_verify, args.batch_size, combined=True, existence=False, return_thresh = True, label_vals_verify = label_vals_verify, return_minmax=True)
            mean_acc = meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify, y_verify)
            min_test, max_test = minmax_vals
            assert(min_test <= max_test)

            with open(os.path.join(plots_path, "ft_rates.npy"), 'wb') as f:
                for arr in ft_rates:
                    np.save(f, arr)
            print(f'FAR/TAR/FRR/TRR at {os.path.join(classifier_save_path, "ft_rates.npy")}')

            torch.save({'rep_gen_state_dicts': [best_representation_generator.state_dict()],
                        'thresholds':[thresh],
                        'minmax_vals':[minmax_vals]}, os.path.join(classifier_save_path, f'{str(round(best_acc,2))}_{str(mean_acc)}.pt'))

        else:
            assert('Ensemble CENSUS not supported')



    elif args.dataset == "BONEAGE":
        result_path = result_path / "BONEAGE"

        targets = ["0.2", "0.3","0.4", "0.5", "0.6", "0.7", "0.8"]
        targets = sorted(targets)
        
        new_targets, targets = attestation_dataset_utils.create_targets(targets, args, window_size=args.window_size, align=args.window_alignment)

        X_train, Y_train, X_test, Y_test, dims, _ = attestation_dataset_utils.assemble_attestation_train_val_data(args, targets, new_targets, result_path)
        X_test = attestation_dataset_utils.batch_boneage(X_test)
        X_train = attestation_dataset_utils.batch_boneage(X_train)

        if args.do_defense:
            X_train, Y_train, X_test, Y_test = robustness.prepare_defense_data(args, adv_samples_path, X_train, Y_train, X_test, Y_test)


        
        best_representation_generator, best_acc = train_attestation_model(args, [X_train], Y_train, [X_test], Y_test, dims)

        X_verify, Y_verify, label_vals_verify, _ =  attestation_dataset_utils.assemble_attestation_test_data(args, targets, new_targets, result_path, return_target_ratio_vals=True)
        X_verify = attestation_dataset_utils.batch_boneage(X_verify)

        y_pred_test, y_test, y_pred_verify, y_verify, thresh , ft_rates, minmax_vals = meta_utils.test_meta_threshold(args, plots_path, best_representation_generator, [X_test], Y_test, [X_verify], Y_verify, args.batch_size, combined = True, existence=False, return_thresh = True, label_vals_verify=label_vals_verify, return_minmax=True)
        mean_acc = meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify, y_verify)
        min_test, max_test = minmax_vals
        assert(min_test <= max_test)

        torch.save({'rep_gen_state_dicts': [best_representation_generator.state_dict()],
                        'thresholds':[thresh],
                        'minmax_vals':[minmax_vals]}, os.path.join(classifier_save_path, f'{str(round(best_acc,2))}_{str(mean_acc)}.pt'))
        
        with open(os.path.join(plots_path, "ft_rates.npy"), 'wb') as f:
            for arr in ft_rates:
                np.save(f, arr)
            print(f'FAR/TAR/FRR/TRR at {os.path.join(classifier_save_path, "ft_rates.npy")}')


    elif args.dataset == "ARXIV":
        result_path = result_path / "ARXIV"
        targets = ["9", "10", "11", "12", "13", "14", "15", "16", "17"]

        ds = data.ArxivNodeDataset('verifier')

        '''targets.remove(args.claim_degree)'''
        new_targets, targets = attestation_dataset_utils.create_targets(targets, args, window_size=args.window_size, align=args.window_alignment)
        X_train, Y_train, X_test, Y_test, dims, _ = attestation_dataset_utils.assemble_attestation_train_val_data(args, targets, new_targets, result_path, ds=ds)

        if args.do_defense:
            X_train, Y_train, X_test, Y_test = robustness.prepare_defense_data(args, adv_samples_path, X_train, Y_train, X_test, Y_test)
        best_representation_generator, best_acc = train_attestation_model(args, X_train, Y_train, X_test, Y_test, dims)

        X_verify, Y_verify, label_vals_verify, _ =  attestation_dataset_utils.assemble_attestation_test_data(args, targets, new_targets, result_path, ds=ds, return_target_ratio_vals = True)
        y_pred_test, y_test, y_pred_verify, y_verify, thresh , ft_rates, minmax_vals = meta_utils.test_meta_threshold(args, plots_path, best_representation_generator, X_test, Y_test, X_verify, Y_verify, args.batch_size, combined=True, existence=False, return_thresh = True, label_vals_verify = label_vals_verify, return_minmax=True)
        mean_acc = meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify, y_verify)
        min_test, max_test = minmax_vals
        assert(min_test <= max_test)

        torch.save({'rep_gen_state_dicts': [best_representation_generator.state_dict()],
                        'thresholds':[thresh],
                        'minmax_vals':[minmax_vals]}, os.path.join(classifier_save_path, f'{str(round(best_acc,2))}_{str(mean_acc)}.pt'))
        with open(os.path.join(plots_path, "ft_rates.npy"), 'wb') as f:
            for arr in ft_rates:
                np.save(f, arr)
            print(f'FAR/TAR/FRR/TRR at {os.path.join(classifier_save_path, "ft_rates.npy")}')
        
    else:
        msg_dataset_error: str = "No such dataset"
        log.error(msg_dataset_error)
        raise EnvironmentError(msg_dataset_error)


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--plots_path', type=str, default='plots/', help="Directory to store the plots")
    parser.add_argument('--result_path', type=str, default="results/", help='Directory where the base models are stored')
    parser.add_argument('--classifier_save_path', type=str, default='test/', help='Save directory for the attestation classifiers')
    parser.add_argument('--classifier_save_suffix', type=str, default='', help="Suffix for the specific directory in which this particular dataset's attestation classifier will be saved")
    parser.add_argument('--adv_samples_path', type=str, default='adv_samples_bb/', help = 'Adversarial samples are stored here')
    parser.add_argument('--do_defense', action='store_true', default=False, help = 'Conduct adversarial training defense for attestation classifier')
    parser.add_argument('--train_prover_version', action='store_true', default=False, help="Train the prover's attestation classifier (swap the splits)")
    parser.add_argument('--dataset', type=str, default="CENSUS", help='Options: CENSUS, BONEAGE')
    parser.add_argument('--filter', type=str,required=False,help='which filter to use')
    parser.add_argument('--verify_base', action='store_true', required=False, default=False, help='toggle to decide whether or not to verify the base model')
    parser.add_argument('--window_size', type=int, default = 1, required=False, help='window size for pclaim')
    parser.add_argument('--window_alignment', type=str, default='centre', required=False, help='if the window is centred on pclaim or only left or right is taken')
    parser.add_argument('--split', choices=["verifier", "prover"], required=False, default = 'prover', help='which split of data to use. This is used for dataset analysis only')
    parser.add_argument('--use_single_attest', action='store_true', required=False, default=True, help='Toggle to force Census attestation to use just one attestation classifier')
    parser.add_argument('--use_more_adj_samples', action='store_true', required=False, default=False, help='Toggle to force Census attestation to use more training samples from target ratios closer to claim ratio')

    parser.add_argument('--train_sample', type=int, default=800,help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=0,help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=10000)

    parser.add_argument('--epochs', type=int, default=1500,help="Number of epochs to train meta-classifier") # Sex: 1000 epochs, 1e-3 Race: 500* epochs, 1e-3
    parser.add_argument('--start_n', type=int, default=0,help="Only consider starting from this layer")
    parser.add_argument('--first_n', type=int, default=1,help="Use only first N layers' parameters")
    parser.add_argument('--ntimes', type=int, default=10,help='number of repetitions for multimode')
    parser.add_argument('--shift', type=int, default=0,help='rotate list')

    parser.add_argument('--epochs_boneage', type=int, default=500)
    parser.add_argument('--claim_ratio', type=str, help="Ratio for D_0", default="0.5")
    parser.add_argument('--prune_ratio', type=float, default=None,help="Prune models before training meta-models [BONEAGE]")

    # Arxiv specific args
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size_arxiv', type=int, default=256)
    parser.add_argument('--epochs_arxiv', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--claim_degree', default="12")
    # parser.add_argument('--regression', action="store_true")
    # parser.add_argument('--gpu', action="store_true")
    # parser.add_argument('--parallel', action="store_true")
    parser.add_argument('--prune', type=float, default=0)

    parser.add_argument('--start_n_conv', type=int, default=0,
                        help="Only consider starting from this layer of conv part")
    parser.add_argument('--first_n_conv', type=int, default=1,
                        help="Only consider first N layers of conv part")
    parser.add_argument('--conv_custom', default=None,
                        help="Comma-separated list of layers wanted (overrides first/last N) for Conv")
    parser.add_argument('--start_n_fc', type=int, default=0,
                        help="Only consider starting from this layer of fc part")
    parser.add_argument('--first_n_fc', type=int, default=1,
                        help="Only consider first N layers of fc part")
    parser.add_argument('--fc_custom', default=None,
                        help="Comma-separated list of layers wanted (overrides first/last N) for FC")
    parser.add_argument('--focus', choices=["fc", "conv", "all", "combined"],
                        default="combined", help="Which layer paramters to use")                        
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="property_attestation.log", filemode="w")
    log: logging.Logger = logging.getLogger("PropertyAttestation")
    args = handle_args()
    
    # execute code segment and find the execution time
    t = Timer(lambda: main(args, log))
    exec_time = t.timeit(number=1)
    
    # printing the execution time in secs.
    # nearest to 2-decimal places
    print(f"Total attestation classifier training time: {exec_time:.02f} secs.")
