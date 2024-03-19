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
from sklearn.utils import shuffle
import random
from tqdm import tqdm
from datetime import datetime
from . import meta_utils
from . import os_layer
from . import models
from . import data
from . import attestation_dataset_utils
from . import robustness
from . import train_verifier_model
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.ensemble import StackingClassifier
import pprint
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_saved_representation_generators(args, folder_path, dims):
    models_in_folder = os.listdir(folder_path)#[:int(args.nmodels)]
    rep_gens = []
    attestation_threshes = []
    minmax_vals_to_ret = []

    # Picks the best model
    best_thresh_acc = 0
    model_to_choose = ""
    for path in tqdm(models_in_folder):
        name_paths = path.split('_')
        rep_gen_acc = float(name_paths[0])
        thresh_acc = float(name_paths[-1][:-3])
        if thresh_acc >= best_thresh_acc:
            best_thresh_acc = thresh_acc
            model_to_choose = path
    print(f'Choosing file: {os.path.join(folder_path, model_to_choose)}')
    model_dict = torch.load(os.path.join(folder_path, model_to_choose))
    
    # There could be multiple model state_dicts saved (as in the case of Census)
    for i, md in enumerate(model_dict['rep_gen_state_dicts']):
        rep_gen = models.GenerateRepresentation2(dims).to(device)
        rep_gen_state_dict = md
        rep_gen.load_state_dict(rep_gen_state_dict)
        attestation_thresh = model_dict['thresholds'][i]
        try:
            minmax_vals = model_dict['minmax_vals'][i]
            minmax_vals_to_ret.append(minmax_vals)
        except:
            print('Model does not have minmax_vals saved. Will return empty list for these values')
        rep_gens.append(rep_gen)
        attestation_threshes.append(attestation_thresh)
    
    if args.dataset != 'CENSUS' and len(rep_gens) == 1:
        return rep_gens[0], attestation_threshes[0], minmax_vals_to_ret
    return rep_gens, attestation_threshes, minmax_vals_to_ret


def main(args, log):
    raw_path: Path = Path(args.raw_path)
    result_path: Path = Path(args.result_path)
    # result_path2: Path = Path(args.result_path2)
    adv_path: Path = Path(args.adv_basemodels_path)
    adv_samples_path: Path = Path(args.adv_samples_path)
    classifier_save_path: Path = Path(args.classifier_save_path)
    classifier_dir_name = args.dataset + args.classifier_suffix
    classifier_dir_name_prover = args.dataset + args.classifier_suffix_prover
    dataset_dir_name = args.dataset + args.dataset_suffix
    if args.dataset == "CENSUS":
        plots_path: Path = Path(args.plots_path) / dataset_dir_name / "Property Attestation" / args.filter / args.claim_ratio
        classifier_save_path_prover = classifier_save_path / classifier_dir_name_prover / args.filter / args.claim_ratio
        classifier_save_path = classifier_save_path / classifier_dir_name / args.filter / args.claim_ratio
    else:
        plots_path: Path = Path(args.plots_path) / dataset_dir_name / "Property Attestation" / args.claim_ratio
        if args.dataset == 'ARXIV':
            plots_path: Path = Path(args.plots_path) / dataset_dir_name / "Property Attestation" / args.claim_degree
            classifier_save_path_prover = classifier_save_path / classifier_dir_name_prover / args.claim_degree
            classifier_save_path = classifier_save_path / classifier_dir_name / args.claim_degree
        else:
            classifier_save_path_prover = classifier_save_path / classifier_dir_name_prover / args.claim_ratio
            classifier_save_path = classifier_save_path / classifier_dir_name / args.claim_ratio

    plots_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(plots_path, log)
    adv_samples_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(adv_samples_path, log)
    adv_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(adv_path, log)
    if plots_path_status is None or adv_samples_path_status is None or adv_path_status is None:
        msg: str = f"Something went wrong when creating {plots_path} or {adv_samples_path} or {adv_path}. Aborting..."
        log.error(msg)
        raise EnvironmentError(msg)
    

    if args.dataset == "CENSUS":
        raw_path = raw_path / "census_splits"
        result_path = result_path / "CENSUS"
        adv_path = adv_path / "CENSUS"
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

        if args.claim_ratio == "0.0" or args.claim_ratio == "1.0":
                args.window_size = 0
        if args.filter == 'race' and (args.claim_ratio == '0.1' or args.claim_ratio == '0.9'):
            args.window_size = 0
            
        # if float(args.claim_ratio) >= 0.3 and float(args.claim_ratio) <= 0.7:
        #         args.window_size = 2
                # args.window_alignment = 'right'

        new_targets, targets = attestation_dataset_utils.create_targets(targets, args, window_size=args.window_size, align=args.window_alignment)
        
        X_train, Y_train, X_test, Y_test, dims, _ = attestation_dataset_utils.assemble_attestation_train_val_data(args, targets, new_targets, result_path, more_adj_samples=args.use_more_adj_samples)
        params_verify, labels_verify, label_vals_verify, clfs = attestation_dataset_utils.assemble_attestation_test_data(args, targets, new_targets, result_path, return_target_ratio_vals = True)
        print(f'Number of actual positive: {labels_verify.sum()}')
        print(f'Number of actual negative: {len(labels_verify) - labels_verify.sum()}')

        print(classifier_save_path)
        rep_gen, attestation_thresh, minmax_vals_saved = get_saved_representation_generators(args, classifier_save_path, dims)
        print(f'minmax_vals_saved: {minmax_vals_saved}')
        print(attestation_thresh)
        y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates, minmax_vals = meta_utils.test_meta_threshold(args, plots_path, rep_gen[0], X_test, Y_test, params_verify, labels_verify, args.batch_size, combined=True, existence=False, return_thresh = True, return_minmax=True)
        mean_acc = meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify, y_verify, args=args, plots_path=plots_path)
        if len(minmax_vals_saved) > 0:
            min_test, max_test = minmax_vals_saved[0]
        else:
            min_test, max_test = minmax_vals
        assert(min_test <= max_test)
        attestation_thresh[0] = attestation_thresh[0] * (max_test - min_test) + min_test

        # print(f'len params_verify: {len(params_verify)}') == 1
        params_verify_robustness = params_verify[0]
        print(f'params_verify shape: {params_verify_robustness.shape}')
        
        if args.do_attack:
            if not args.untargeted_attack:
                print(f'labels_verify type: {type(labels_verify)}')
                print(f'labels_verify: {labels_verify}')
                labels_verify_adv = torch.zeros_like(labels_verify)
            else:
                labels_verify_adv = labels_verify
            rep_gen_to_attack = rep_gen
            attestation_thresh_to_attack = attestation_thresh
            if args.bb_adv_attack:
                rep_gen_prover, attestation_thresh_prover, minmax_vals_saved_prover = get_saved_representation_generators(args, classifier_save_path_prover, dims)
                if len(minmax_vals_saved_prover) > 0:
                    min_test_prover, max_test_prover = minmax_vals_saved_prover[0]
                    assert(min_test_prover <= max_test_prover)
                else:
                    min_test_prover, max_test_prover = min_test, max_test
                attestation_thresh_prover[0] = attestation_thresh_prover[0] * (max_test_prover - min_test_prover) + min_test_prover
                rep_gen_to_attack = rep_gen_prover
                attestation_thresh_to_attack = attestation_thresh_prover
                
            # Robustness only supports the non-ensemble creation of census
            x_advs = robustness.do_attack(args, rep_gen_to_attack[0], params_verify_robustness, eps = 8/255, y = labels_verify_adv, attestation_thresh = attestation_thresh_to_attack[0], actual_ratio_labels=label_vals_verify)
            
            x_advs_verify_squeezed = []
            for li in x_advs:
                x_advs_verify_squeezed += li

            if not args.untargeted_attack:
                x_advs_verify_squeezed_tensor = torch.cat(x_advs_verify_squeezed)
                x_advs_verify_targets = x_advs_verify_squeezed_tensor[labels_verify==0]
                x_verify_pos = params_verify_robustness[labels_verify==1]
                x_advs = [torch.cat((x_advs_verify_targets, x_verify_pos))]
            else:
                x_advs = [torch.cat(x_advs, dim=0).squeeze(1)]

        if args.do_defense:
            X_train_adv = X_train[0]
            X_test = X_test[0]
            robustness.generate_and_save_perturbed_data(args, rep_gen[0], attestation_thresh[0], X_train_adv, Y_train, X_test, Y_test, adv_samples_path, log)
            print('Perturbed data saved successfully')
            
        if len(attestation_thresh) > 1:
            y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates = meta_utils.test_meta_threshold(args, plots_path, rep_gen, X_test, Y_test, params_verify, labels_verify, args.batch_size, threshes = attestation_thresh, combined=True, existence=False, return_thresh = True, label_vals_verify = label_vals_verify)
        else:
            print()
            print('Performance on clean set')
            rep_gen[0].aucroc_thresh = attestation_thresh[0]
            y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates = meta_utils.test_meta_threshold(args, plots_path, rep_gen[0], X_test, Y_test, params_verify, labels_verify, args.batch_size, combined=True, existence=False, return_thresh = True, label_vals_verify = label_vals_verify)
            mean_acc = meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify.detach().cpu().numpy(), y_verify.detach().cpu().numpy(), args=args, plots_path=plots_path)

            if args.do_attack:
                rep_gen[0].aucroc_thresh = attestation_thresh[0]
                print()
                print('Performance on perturbed set')
                # Let's see the performance degradation of the attestation classifier
                y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates = meta_utils.test_meta_threshold(args, plots_path, rep_gen[0], X_test, Y_test, x_advs, labels_verify, args.batch_size, combined = True, existence=False, return_thresh = True, label_vals_verify = label_vals_verify)
                mean_acc = meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify.detach().cpu().numpy(), y_verify.detach().cpu().numpy(), args=args, plots_path=plots_path)
                print()

                if args.check_finetuned_accuracy:
                    base_models_to_check = robustness.choose_basemodels(args, x_advs_verify_squeezed, label_vals_verify, clfs)
                    attacked_models = robustness.check_basemodel_performance_on_perturbed(args, base_models_to_check, result_path, adv_path)
                


    elif args.dataset == 'BONEAGE':
        raw_path = raw_path / "boneage_splits"
        result_path = result_path / "BONEAGE"
        adv_path = adv_path / "BONEAGE"

        targets = ["0.2", "0.3","0.4", "0.5", "0.6", "0.7", "0.8"]
        targets = sorted(targets)
        
        new_targets, targets = attestation_dataset_utils.create_targets(targets, args, window_size=args.window_size, align=args.window_alignment)

        X_train, Y_train, X_test, Y_test, dims, _ = attestation_dataset_utils.assemble_attestation_train_val_data(args, targets, new_targets, result_path)
        X_verify, Y_verify, label_vals_verify, clfs =  attestation_dataset_utils.assemble_attestation_test_data(args, targets, new_targets, result_path, return_target_ratio_vals = True)
        print(f'Number of actual positive: {Y_verify.sum()}')
        print(f'Number of actual negative: {len(Y_verify) - Y_verify.sum()}')

        X_test = attestation_dataset_utils.batch_boneage(X_test).to(device)
        X_verify = attestation_dataset_utils.batch_boneage(X_verify).to(device)
        # print(f'X_test shape: {X_test.shape}')

        rep_gen, attestation_thresh, minmax_vals_saved = get_saved_representation_generators(args, classifier_save_path, dims)
        print(f'minmax_vals_saved: {minmax_vals_saved}')
        y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates, minmax_vals = meta_utils.test_meta_threshold(args, plots_path, rep_gen, [X_test], Y_test, [X_verify], Y_verify, args.batch_size, combined = True, existence=False, return_thresh = True, return_minmax=True)
        meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify, y_verify, args=args, plots_path=plots_path)
        print()
        if len(minmax_vals_saved) > 0:
            min_test, max_test = minmax_vals_saved[0]
        else:
            min_test, max_test = minmax_vals
        assert(min_test <= max_test)
        attestation_thresh = attestation_thresh * (max_test - min_test) + min_test
        
        if args.do_attack:
            if not args.untargeted_attack:
                print(f'Y_verify: {Y_verify}')
                Y_verify_adv = torch.zeros_like(Y_verify)
            else:
                Y_verify_adv = Y_verify
            rep_gen_to_attack = rep_gen
            attestation_thresh_to_attack = attestation_thresh
            print(f'attestation thresh: {attestation_thresh}')

            # rep_gen_prover is the prover's rep_gen (prover has whitebox access to this, and attacks it)
            if args.bb_adv_attack:
                rep_gen_prover, attestation_thresh_prover, minmax_vals_saved_prover = get_saved_representation_generators(args, classifier_save_path_prover, dims)
                if len(minmax_vals_saved_prover) > 0:
                    min_test_prover, max_test_prover = minmax_vals_saved_prover[0]
                    assert(min_test_prover <= max_test_prover)
                else:
                    min_test_prover, max_test_prover = min_test, max_test
                attestation_thresh_prover = attestation_thresh_prover * (max_test_prover - min_test_prover)
                rep_gen_to_attack = rep_gen_prover
                attestation_thresh_to_attack = attestation_thresh_prover
                print(f'attestation thresh prover: {attestation_thresh_prover}')
            
            start = time.time()
            x_advs_verify = robustness.do_attack(args, rep_gen_to_attack.cuda(), X_verify, eps = 8/255, y = Y_verify_adv, attestation_thresh = attestation_thresh_to_attack)
            end = time.time()
            print(f'Time taken to conduct attack: {end-start}')
            
            x_advs_verify_squeezed = []
            for li in x_advs_verify:
                x_advs_verify_squeezed += li
            
            print(f'X_verify shape: {X_verify.shape}, Y_verify shape: {Y_verify.shape}, len(clfs): {len(clfs)}, len(label_vals_verify): {len(label_vals_verify)}')
            print(f'len(x_advs_verify_squeezed): {len(x_advs_verify_squeezed)}')

            # We basically did the above in the following comment block too
            if not args.untargeted_attack:
                x_advs_verify_squeezed_tensor = torch.cat(x_advs_verify_squeezed)
                x_advs_verify_targets = x_advs_verify_squeezed_tensor[Y_verify==0]
                x_verify_pos = X_verify[Y_verify==1]
                x_advs_verify = torch.cat((x_advs_verify_targets.cuda(), x_verify_pos))
                print(f'Targeted attack')
            else:
                x_advs_verify = torch.cat(x_advs_verify, dim=0).squeeze(1)


        if args.do_defense:
            X_train_adv = attestation_dataset_utils.batch_boneage(X_train).to(device)

            # Verifier will be the one that is doing adversarial training, so use verifier's rep_gen to generate adversarial samples
            robustness.generate_and_save_perturbed_data(args, rep_gen, attestation_thresh, X_train_adv, Y_train, X_test, Y_test, adv_samples_path, log)
            print('Perturbed data saved successfully')

        # Rep_gen_to_use is always the verifier's rep_gen
        rep_gen_to_use = rep_gen
        x_test_to_use = X_test
        y_test_to_use = Y_test
        


        print()
        print('Performance on clean set')
        rep_gen_to_use.aucroc_thresh = None
        # if args.test_defended:
        rep_gen_to_use.aucroc_thresh = attestation_thresh
        y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates = meta_utils.test_meta_threshold(args, plots_path, rep_gen_to_use, [x_test_to_use], y_test_to_use, [X_verify], Y_verify, args.batch_size, combined = True, existence=False, return_thresh = True, label_vals_verify=label_vals_verify)
        try:
            meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify.detach().cpu().numpy(), y_verify.detach().cpu().numpy(), args=args, plots_path=plots_path)
        except:
            meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify, y_verify, args=args, plots_path=plots_path)
        print(f'thresh: {thresh}')
        
        if args.do_attack:
            rep_gen_to_use.aucroc_thresh = attestation_thresh

            print()
            print('Performance on perturbed set')
            # Let's see the performance degradation of the attestation classifier
            y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates = meta_utils.test_meta_threshold(args, plots_path, rep_gen_to_use, [x_test_to_use], y_test_to_use, [x_advs_verify], Y_verify, args.batch_size, combined = True, existence=False, return_thresh = True, label_vals_verify=label_vals_verify)

            meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify.detach().cpu().numpy(), y_verify.detach().cpu().numpy(), args=args, plots_path=plots_path)
            print()

            if args.check_finetuned_accuracy:
                # Because this takes so long, do the finetuned verification after the attestation evaluation
                base_models_to_check = robustness.choose_basemodels(args, x_advs_verify_squeezed, label_vals_verify, clfs)
                attacked_models = robustness.check_basemodel_performance_on_perturbed(args, base_models_to_check, result_path, adv_path)
            

    
    elif args.dataset == 'ARXIV':
        result_path = result_path / "ARXIV"
        adv_path = adv_path / "ARXIV"
        targets = ["9", "10", "11", "12", "13", "14", "15", "16", "17"]

        ds = data.ArxivNodeDataset('verifier')

        '''targets.remove(args.claim_degree)'''
        new_targets, targets = attestation_dataset_utils.create_targets(targets, args, window_size=args.window_size, align=args.window_alignment)
        X_train, Y_train, X_test, Y_test, dims, _ = attestation_dataset_utils.assemble_attestation_train_val_data(args, targets, new_targets, result_path, ds=ds)
        X_verify, Y_verify, label_vals_verify, clfs =  attestation_dataset_utils.assemble_attestation_test_data(args, targets, new_targets, result_path, ds=ds, return_target_ratio_vals = True)
        print(f'Number of actual positive: {Y_verify.sum()}')
        print(f'Number of actual negative: {len(Y_verify) - Y_verify.sum()}')
        
        rep_gen, attestation_thresh, minmax_vals_saved = get_saved_representation_generators(args, classifier_save_path, dims)
        print(f'minmax_vals_saved: {minmax_vals_saved}')
        y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates, minmax_vals = meta_utils.test_meta_threshold(args, plots_path, rep_gen, X_test, Y_test, X_verify, Y_verify, args.batch_size, combined = True, existence=False, return_thresh = True, return_minmax = True)
        meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify, y_verify, args=args, plots_path=plots_path)
        if len(minmax_vals_saved) > 0:
            min_test, max_test = minmax_vals_saved[0]
        else:
            min_test, max_test = minmax_vals
        assert(min_test <= max_test)
        attestation_thresh = attestation_thresh * (max_test - min_test) + min_test

        if args.do_attack:
            if not args.untargeted_attack:
                print('Targeted')
                print(f'Y_verify: {Y_verify}')
                Y_verify_adv = torch.zeros_like(Y_verify)
            else:
                Y_verify_adv = Y_verify
            rep_gen_to_attack = rep_gen
            attestation_thresh_to_attack = attestation_thresh
            if args.bb_adv_attack:
                rep_gen_prover, attestation_thresh_prover, minmax_vals_saved_prover = get_saved_representation_generators(args, classifier_save_path_prover, dims)
                if len(minmax_vals_saved_prover) > 0:
                    min_test_prover, max_test_prover = minmax_vals_saved_prover[0]
                    assert(min_test_prover <= max_test_prover)
                else:
                    min_test_prover, max_test_prover = min_test, max_test
                attestation_thresh_prover = attestation_thresh_prover * (max_test_prover - min_test_prover)
                rep_gen_to_attack = rep_gen_prover
                attestation_thresh_to_attack = attestation_thresh_prover

            X_verify_robustness = X_verify[0]

            x_advs = robustness.do_attack(args, rep_gen_to_attack, X_verify_robustness, eps = 8/255, y = Y_verify_adv, attestation_thresh = attestation_thresh_to_attack, actual_ratio_labels=label_vals_verify)

            x_advs_verify_squeezed = []
            for li in x_advs:
                x_advs_verify_squeezed += li

            # We basically did the above in the following comment block too
            if not args.untargeted_attack:
                print(f'Y_verify: {Y_verify} with shape {Y_verify.shape}')
                x_advs_verify_squeezed_tensor = torch.cat(x_advs_verify_squeezed)
                print(f'x_advs_verify_squeezed_tensor: {x_advs_verify_squeezed_tensor.shape}')
                x_advs_verify_targets = x_advs_verify_squeezed_tensor[Y_verify==0.]
                x_verify_pos = X_verify_robustness[Y_verify.cpu()==1.]
                x_advs = [torch.cat((x_verify_pos.cuda(), x_advs_verify_targets.cuda()))]
            else:
                x_advs = [torch.cat(x_advs, dim=0).squeeze(1)]


        if args.do_defense:
            X_train_adv = X_train[0]
            X_test = X_test[0]

            # Verifier will be the one that is doing adversarial training, so use verifier's rep_gen to generate adversarial samples
            robustness.generate_and_save_perturbed_data(args, rep_gen, attestation_thresh, X_train_adv, Y_train, X_test, Y_test, adv_samples_path, log)
            print('Perturbed data saved successfully')

        # print('Going to test')
        print()
        print('Performance on clean set')
        # if args.test_defended:
        rep_gen.aucroc_thresh = attestation_thresh
        y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates = meta_utils.test_meta_threshold(args, plots_path, rep_gen, X_test, Y_test, X_verify, Y_verify, args.batch_size, combined=True, existence=False, return_thresh = True, label_vals_verify = label_vals_verify)
        meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify.detach().cpu().numpy(), y_verify.detach().cpu().numpy(), args=args, plots_path=plots_path)
        rep_gen.aucroc_thresh = attestation_thresh
        print(f'thresh: {thresh}')

        if args.do_attack:
            # Let's see the performance degradation of the attestation classifier
            print()
            print('Performance on perturbed set')
            y_pred_test, y_test, y_pred_verify, y_verify, thresh, ft_rates = meta_utils.test_meta_threshold(args, plots_path, rep_gen, X_test, Y_test, x_advs, Y_verify, args.batch_size, combined = True, existence=False, return_thresh = True, label_vals_verify=label_vals_verify)
            meta_utils.compute_metrics(y_pred_test, y_test, y_pred_verify.detach().cpu().numpy(), y_verify.detach().cpu().numpy(), args=args, plots_path=plots_path)
            print(f'thresh: {thresh}')
            print()
            
            
            if args.check_finetuned_accuracy:
                base_models_to_check = robustness.choose_basemodels(args, x_advs_verify_squeezed, label_vals_verify, clfs)
                attacked_models = robustness.check_basemodel_performance_on_perturbed(args, base_models_to_check, result_path, adv_path)
            

    # with open(os.path.join(plots_path, "ft_rates.npy"), 'wb') as f:
    #     for arr in ft_rates:
    #         np.save(f, arr)
    # print(f'FAR/TAR/FRR/TRR at {os.path.join(classifier_save_path, "ft_rates.npy")}')


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default="dataset/", help='Root directory of the dataset')
    parser.add_argument('--plots_path', type=str, default='plots/', help="Directory to store the plots")
    parser.add_argument('--result_path', type=str, default="results/", help='Results directory for the models')
    # parser.add_argument('--result_path', type=str, default="models_KL/models_boneage_nobalancing/bonemodel/normal", help='Results directory for the models')
    # parser.add_argument('--adv_basemodels_path', type=str, default='results_adv_KL/', help='Results directory for the finetuned adversarial base models')
    parser.add_argument('--adv_basemodels_path', type=str, default='results_adv_bb/', help='Results directory for the finetuned adversarial base models')
    parser.add_argument('--adv_samples_path', type=str, default='adv_samples_bb/', help = 'Adversarial samples will be stored here')
    parser.add_argument('--classifier_save_path', type=str, default='test/', help='Save directory for the attestation classifiers')
    parser.add_argument('--classifier_suffix', type=str, default='', help='CENSUS, BONEAGE, or ARXIV classifier dir could have some suffix at the end for their directory')
    parser.add_argument('--classifier_suffix_prover', type=str, default='', help='CENSUS, BONEAGE, or ARXIV blackbox classifier dir could have some suffix at the end for their directory')
    parser.add_argument('--bb_adv_attack', default = False, action='store_true', help='Toggle to conduct blackbox adversarial attack')
    parser.add_argument('--dataset', type=str, default="CENSUS", help='Options: CENSUS, BONEAGE')
    parser.add_argument('--dataset_suffix', type=str, default='', help='CENSUS, BONEAGE, or ARXIV dataset could have some suffix at the end for their plots directory') 
    parser.add_argument('--filter', type=str,required=False,help='which filter to use')
    parser.add_argument('--verify_base', action='store_true', required=False, default=False, help='toggle to decide whether or not to verify the base model')
    parser.add_argument('--window_size', type=int, default = 1, required=False, help='window size for pclaim')
    parser.add_argument('--window_alignment', type=str, default='centre', required=False, help='if the window is centred on pclaim or only left or right is taken')
    parser.add_argument('--use_single_attest', action='store_true', required=False, default=False, help='Toggle to force Census attestation to use just one attestation classifier')
    parser.add_argument('--use_more_adj_samples', action='store_true', required=False, default=False, help='Toggle to force Census attestation to use more training samples from target ratios closer to claim ratio. (Not really needed here)')
    parser.add_argument('--do_not_plot_roc', action='store_true', default=False, help='Toggle to force the ROC curve plot not to get saved')
    parser.add_argument('--do_attack', action='store_true', default=False, help = 'Conduct perturbation-based attack on attestation classifier')
    parser.add_argument('--untargeted_attack', action='store_true', default = False, help = 'Do untargeted perturbation-based attack')
    parser.add_argument('--do_defense', action='store_true', default=False, help = 'Conduct adversarial training defense for attestation classifier; ie. store the perturbed traiining and test data')
    parser.add_argument('--test_defended', action='store_true', default=False, help = 'Test the adversarially defended verifier attestation model')
    parser.add_argument('--check_finetuned_accuracy', action='store_true', default=False, help='Check the accuracy of a basemodel after it has been finetuned with adversarial first layer weights')

    # Just to use the scripts to get validation data, the train_sample argument is taken here
    parser.add_argument('--train_sample', type=int, default=800,help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=0,help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=10000)

    parser.add_argument('--start_n', type=int, default=0,help="Only consider starting from this layer")
    parser.add_argument('--first_n', type=int, default=1,help="Use only first N layers' parameters")
    parser.add_argument('--ntimes', type=int, default=1,help='number of repetitions for multimode')
    parser.add_argument('--nmodels', type=int, default=1,help='number of models to take from directory')
    parser.add_argument('--num_finetuned_basemodels', type=int, default = 10, help='The number of finetuned base models to verify the performance of for each ratio')

    parser.add_argument('--prune_ratio', type=float, default=None,help="Prune models before training meta-models [BONEAGE]")
    parser.add_argument('--epochs_boneage', type=int, default=500)
   
    # Arxiv specific args
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size_arxiv', type=int, default=256)
    parser.add_argument('--epochs_arxiv', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--prune', type=float, default=0)

    parser.add_argument('--claim_ratio', type=str, help="Ratio for D_0", default="0.5")
    parser.add_argument('--claim_degree', default="12")
                      
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="verification.log", filemode="w")
    log: logging.Logger = logging.getLogger("Verification")
    args = handle_args()
    # Make sure that in a blackbox attack, different attestation classifiers are being used
    if args.bb_adv_attack:
        assert(args.classifier_suffix != args.classifier_suffix_prover)
    main(args, log)
