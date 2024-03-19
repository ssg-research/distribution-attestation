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
import os
import numpy as np
import torch
from pathlib import Path
from typing import List
import scipy.stats
from sklearn.utils import shuffle
import random
from . import meta_utils
from . import os_layer
from . import models
from . import data
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_list_full_length(the_list):
    total_len = 0
    for index, li in enumerate(the_list):
        total_len += len(li)
        print(len(li))
    return total_len


def adjust_dataset_lengths(list_list_w, list_list_labels = None, num_to_take = None):
    for idx, li in enumerate(list_list_w):
        # Boneage will send no list_list_labels
        if list_list_labels is not None:
            li, list_list_labels[idx] = shuffle(li, list_list_labels[idx], random_state=0)
        else:
            random.shuffle(li)

        list_list_w[idx] = li[:num_to_take]

        if list_list_labels is not None:
            list_list_labels[idx] = list_list_labels[idx][:num_to_take]
    
    return list_list_labels, list_list_w


def batch_boneage(X_vals):
    comb_x_vals= []
    for li in X_vals:
        comb_x_vals.append(li[0])
    X_vals_new = torch.stack(comb_x_vals)

    return X_vals_new
    

def create_targets(targets, args, window_size = 1, align = 'centre', targets_below_lb = False, targets_above_ub = False):
    new_targets = []
    if args.dataset == 'CENSUS':
        assert(window_size == 0 or window_size == 1 or window_size == 2)
    # Boneage should not allow a window greater than 1 or else there won't be enough ratios on the outside
    elif args.dataset == 'BONEAGE':
        assert(window_size == 0 or window_size == 1)
    if args.dataset == 'CENSUS' or args.dataset == 'BONEAGE':
        claim_ratio_idx = targets.index(args.claim_ratio)
    elif args.dataset == 'ARXIV':
        claim_ratio_idx = targets.index(args.claim_degree)

    lb = claim_ratio_idx-window_size
    ub = claim_ratio_idx+window_size+1
    
    # If at the corner, only take the bound
    if lb < 0:
        lb = 0
    if ub > len(targets):
        ub = len(targets)
    if align == 'right':
        lb = claim_ratio_idx
    elif align == 'left':
        ub = claim_ratio_idx+1

    new_targets = targets[lb:ub]
    if targets_below_lb:
        targets = targets[:lb]
    elif targets_above_ub:
        targets = targets[ub:]
    else:
        targets = targets[:lb] + targets[ub:]
    # targets = targets[ub:]
    print(f'targets: {targets}')
    print(f'new_targets: {new_targets}')
    return new_targets, targets



def assemble_attestation_train_val_data(args, targets, new_targets, result_path, more_adj_samples = False, ds=None, fetch_models = False):
    train_data_split = 'verifier'
    test_data_split = 'prover'
    try:
        if args.train_prover_version:
            train_data_split = 'prover'
            test_data_split = 'verifier'
    except:
        print('--train_prover_version not declared. Ensure that you are running verification, not training')

    if args.dataset == 'CENSUS':
        pos_w_list=[]
        pos_w_test_list=[]
        pos_labels_list=[]
        pos_labels_test_list=[]

        for index, tg in enumerate(new_targets):
            print("Pos Ratio: ", tg)
            pos_w, pos_labels, _, clfs = meta_utils.get_model_representations(meta_utils.get_models_path(result_path,train_data_split, args.filter, tg), args, 1, args.first_n, fetch_models = fetch_models)
            pos_w_test, pos_labels_test, dims, clfs = meta_utils.get_model_representations(meta_utils.get_models_path(result_path,test_data_split, args.filter, tg), args, 1, args.first_n, fetch_models=fetch_models)
            pos_w_list.append(pos_w)
            pos_w_test_list.append(pos_w_test)
            pos_labels_list.append(pos_labels)
            pos_labels_test_list.append(pos_labels_test)

        # Each of the positive lists are len(new_targets) long
        total_len_pos = get_list_full_length(pos_labels_list)
        total_len_pos_test = get_list_full_length(pos_labels_test_list)
        print('Negs next')

        pos_labels_list = np.concatenate(pos_labels_list)
        pos_w_list = np.concatenate(pos_w_list)
        pos_labels_test_list = np.concatenate(pos_labels_test_list)
        pos_w_test_list = np.concatenate(pos_w_test_list)

        neg_labels_list = []
        neg_w_list = []
        neg_labels_test_list = []
        neg_w_test_list = []

        # These two are needed if we are not having a uniform distribution for the target ratios- the ratios closer to claim will
        #    be present in higher proportion
        largest_target = float(targets[-1])
        smallest_target = float(targets[0])
        window_lb = float(new_targets[0])
        window_ub = float(new_targets[-1])

        # The divisor will be different on either side of the window
        # If the largest target is smaller than the window's lower bound, then we have a left window,
        #    and there are no ratios to the right
        if largest_target <= window_lb:
            num_ratios_right = 0
        else:
            num_ratios_right = largest_target*10 - window_ub*10
            divisor_right = num_ratios_right*(num_ratios_right+1)/2
            # highest_distance_to_claim_right = (largest_target - window_ub)*10
        
        # Similarly, if the smallest target is greater than the window's upper bound, then we have a right window,
        #    and there are no ratios to the left
        if smallest_target >= window_ub:
            num_ratios_left = 0
        else:
            num_ratios_left = window_lb*10 - smallest_target*10
            divisor_left = num_ratios_left*(num_ratios_left+1)/2
            # highest_distance_to_claim_left = (window_lb - smallest_target)*10
        
        print(f'Num ratios left: {num_ratios_left}')
        print(f'Num ratios right: {num_ratios_right}')

        # All ratios except the ones in the window are negative labels
        for index, tg in enumerate(targets):
            print("Neg Ratio", tg)
            neg_w, neg_labels, _, clfs = meta_utils.get_model_representations(meta_utils.get_models_path(result_path,train_data_split, args.filter, str(tg)), args, 0, args.first_n, fetch_models=fetch_models)
            neg_w_test, neg_labels_test, _, clfs = meta_utils.get_model_representations(meta_utils.get_models_path(result_path,test_data_split, args.filter, str(tg)), args, 0, args.first_n, fetch_models=fetch_models)
            # if (args.claim_ratio!="0.0") and (args.claim_ratio!="1.0"):
            #     neg_w, neg_labels = neg_w[:375], neg_labels[:375]
            #     neg_w_test, neg_labels_test = neg_w_test[:375], neg_labels_test[:375]
            # else:
            #     neg_w, neg_labels = neg_w[:222], neg_labels[:222]
            #     neg_w_test, neg_labels_test = neg_w_test[:222], neg_labels_test[:222]

            if more_adj_samples:
                distance_to_claim_ratio = float(tg)*10 - float(args.claim_ratio)*10
                # Using a left ratio
                if distance_to_claim_ratio < 0:
                    # Use distance wrt window lower bound
                    distance_to_claim_ratio = float(tg)*10 - window_lb*10
                    distance_to_claim_ratio *= -1
                    # +1 because num_ratios_left and distance_to_claim ratio can be the same (in the case of the corner target ratio)
                    dividend = num_ratios_left - distance_to_claim_ratio + 1
                    divisor = divisor_left
                    bound_used = window_lb
                # Using a right ratio
                else:
                    # Use distance wrt to window upper bound
                    distance_to_claim_ratio = float(tg)*10 - window_ub*10
                    dividend = num_ratios_right - distance_to_claim_ratio + 1
                    divisor = divisor_right
                    bound_used = window_ub
                    
                # dividend = highest_distance_to_claim_ratio - distance_to_claim_ratio
                num_ratio_to_take = int(total_len_pos * dividend // divisor)
                print(f'Taking {num_ratio_to_take} of ratio {tg} with distance {distance_to_claim_ratio}, dividend {dividend}, and divisor {divisor} with bound used {bound_used}')

                neg_w, neg_labels = shuffle(neg_w, neg_labels, random_state=0)
                neg_w_test_list, neg_labels_test_list = shuffle(neg_w_test_list, neg_labels_test_list, random_state=0)

                neg_labels = neg_labels[:num_ratio_to_take]
                neg_w = neg_w[:num_ratio_to_take]
                neg_labels_test_list = neg_labels_test_list[:num_ratio_to_take]
                neg_w_test_list = neg_w_test_list[:num_ratio_to_take]

            neg_labels_list.append(neg_labels)
            neg_w_list.append(neg_w)
            neg_labels_test_list.append(neg_labels_test)
            neg_w_test_list.append(neg_w_test)

        # total_len_neg = get_list_full_length(neg_labels_list)
        # total_len_neg_test = get_list_full_length(neg_labels_test_list)
        
        if not more_adj_samples:
            num_neg_to_take = total_len_pos // len(neg_labels_list)
            num_neg_test_to_take = total_len_pos_test // len(neg_labels_test_list)
            print(f'num_neg_to_take: {num_neg_to_take}, test: {num_neg_test_to_take}')
        
            neg_labels_list, neg_w_list = adjust_dataset_lengths(neg_w_list, neg_labels_list, num_neg_to_take)
            neg_labels_test_list, neg_w_test_list = adjust_dataset_lengths(neg_w_test_list, neg_labels_test_list, num_neg_test_to_take)
            

        neg_labels_list = np.concatenate(neg_labels_list)
        neg_w_list = np.concatenate(neg_w_list)
        neg_labels_test_list = np.concatenate(neg_labels_test_list)
        neg_w_test_list = np.concatenate(neg_w_test_list)

        # Assemble the training and validation datasets for the attestation classifier
        # Unlike the other two datasets, the features are concatenated together in order to have the 
        #   correct shape to join properly (These are 3000 x 1 each)
        X_test = np.concatenate((pos_w_test_list, neg_w_test_list))
        Y_test = torch.cat((torch.from_numpy(pos_labels_test_list), torch.from_numpy(neg_labels_test_list))).to(device)
        X_test = meta_utils.prepare_batched_data(X_test)

        X_train = np.concatenate((pos_w_list, neg_w_list))
        Y_train = np.concatenate((pos_labels_list, neg_labels_list))
        X_train, Y_train = shuffle(X_train, Y_train, random_state = 0)
        Y_train = torch.from_numpy(Y_train)

        Y_train = Y_train.float()
        Y_test = Y_test.float()
        X_train = meta_utils.prepare_batched_data(X_train)
    
    elif args.dataset == 'BONEAGE':
        pos_w_list = []
        pos_w_test_list = []
        for index, tg in enumerate(new_targets):
            print("Pos Ratio: ", tg)
            train_dir_1 = os.path.join(result_path, "%s/%s" % (train_data_split, tg))
            test_dir_1 = os.path.join(result_path, "%s/%s" % (test_data_split, tg))

            dims, vecs_train_1, clfs = meta_utils.get_model_features(train_dir_1, args, first_n=args.first_n,start_n=args.start_n, prune_ratio=args.prune_ratio,max_read=1000, fetch_models=fetch_models)
            _, vecs_test_1, clfs = meta_utils.get_model_features(test_dir_1, args, first_n=args.first_n,start_n=args.start_n, prune_ratio=args.prune_ratio,max_read=1000, fetch_models=fetch_models)

            pos_w_list += vecs_train_1
            pos_w_test_list += vecs_test_1

        vecs_train_pos = np.array(pos_w_list, dtype='object')
        vecs_test_pos = np.array(pos_w_test_list, dtype='object')

        neg_w_list = []
        neg_w_test_list = []
        for index, tg in enumerate(targets):
            print("Neg Ratio", tg)
            train_dir_2 = os.path.join(result_path, "%s/%s" % (train_data_split, tg))
            test_dir_2 = os.path.join(result_path, "%s/%s" % (test_data_split, tg))

            _, vecs_train_2, clfs = meta_utils.get_model_features(train_dir_2, args, first_n=args.first_n,start_n=args.start_n, prune_ratio=args.prune_ratio,max_read=1000, fetch_models=fetch_models)
            _, vecs_test_2, clfs = meta_utils.get_model_features(test_dir_2, args, first_n=args.first_n,start_n=args.start_n, prune_ratio=args.prune_ratio,max_read=1000, fetch_models=fetch_models)
            if (args.claim_ratio == "0.2") or (args.claim_ratio == "0.8"):
                neg_w_list += vecs_train_2[:280]
                neg_w_test_list += vecs_test_2[:280]
            else:
                neg_w_list += vecs_train_2[:524]
                neg_w_test_list += vecs_test_2[:524]

        vecs_train_neg = np.array(neg_w_list, dtype='object')
        vecs_test_neg = np.array(neg_w_test_list, dtype='object')

        Y_test = [0.] * len(vecs_test_neg) + [1.] * len(vecs_test_pos)
        Y_test = torch.from_numpy(np.array(Y_test)).to(device)
        X_test =  np.concatenate((vecs_test_neg, vecs_test_pos))
        X_test = np.array(X_test, dtype='object')

        Y_train = [0.] * len(vecs_train_neg) + [1.] * len(vecs_train_pos)
        Y_train = torch.from_numpy(np.array(Y_train)).to(device)
        X_train = np.concatenate((vecs_train_neg, vecs_train_pos))

        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
    
    elif args.dataset == 'ARXIV':
        train_dirs_pos = [os.path.join(result_path, train_data_split,  x) for x in new_targets]
        test_dirs_pos = [os.path.join(result_path, test_data_split, x) for x in new_targets]

        train_dirs_neg = [os.path.join(result_path, train_data_split,  x) for x in targets]
        test_dirs_neg = [os.path.join(result_path, test_data_split, x) for x in targets]

        X_train_pos, X_test_pos = [], []
        Y_train_pos, Y_test_pos = [], []
        for i, (trd, ted, ratio) in enumerate(zip(train_dirs_pos, test_dirs_pos, new_targets)):
            print("Pos Ratio:", ratio)
            dims, vecs_train, clfs = models.get_model_features_arxiv(trd, ds, args, max_read=args.train_sample + args.val_sample, fetch_models=fetch_models)
            _, vecs_test, clfs = models.get_model_features_arxiv(ted, ds, args, max_read=1000, fetch_models=fetch_models)

            vecs_train = vecs_train[args.val_sample:]

            X_train_pos += vecs_train
            X_test_pos += vecs_test

            Y_train_pos.append([1] * len(vecs_train))
            Y_test_pos.append([1] * len(vecs_test))


        # Interestingly, arxiv also isn't concatenating anything (unlike Census)
        X_train_pos = np.array(X_train_pos, dtype='object')
        X_test_pos = np.array(X_test_pos, dtype='object')
        Y_train_pos = torch.from_numpy(np.concatenate(Y_train_pos)).float().to(device)
        Y_test_pos = torch.from_numpy(np.concatenate(Y_test_pos)).float().to(device)

        X_train_neg, X_test_neg = [], []
        Y_train_neg, Y_test_neg = [], []

        for i, (trd, ted,ratio) in enumerate(zip(train_dirs_neg, test_dirs_neg,targets)):
            print("Neg Ratio: {}".format(ratio))
            dims, vecs_train, clfs = models.get_model_features_arxiv(trd, ds, args, max_read=args.train_sample + args.val_sample, fetch_models=fetch_models)
            _, vecs_test, clfs = models.get_model_features_arxiv(ted, ds, args, max_read=1000, fetch_models=fetch_models)

            X_train_neg += vecs_train[:100]
            X_test_neg += vecs_test[:100]
                
            Y_train_neg.append([0] * len(vecs_train[:100]))
            Y_test_neg.append([0] * len(vecs_test[:100]))

        X_train_neg = np.array(X_train_neg, dtype='object')
        X_test_neg = np.array(X_test_neg, dtype='object')
        Y_train_neg = torch.from_numpy(np.concatenate(Y_train_neg)).float().to(device)
        Y_test_neg = torch.from_numpy(np.concatenate(Y_test_neg)).float().to(device)

        # 800 x 1 each
        X_train = np.concatenate((X_train_pos,X_train_neg))
        X_test = np.concatenate((X_test_pos,X_test_neg))
        Y_train = torch.cat((Y_train_pos,Y_train_neg))
        Y_test = torch.cat((Y_test_pos,Y_test_neg))

        X_train = meta_utils.prepare_batched_data(X_train)
        X_test = meta_utils.prepare_batched_data(X_test)

    return X_train, Y_train, X_test, Y_test, dims, clfs


def assemble_attestation_test_data(args, targets, new_targets, result_path, use_original_targets = False, return_target_ratio_vals = False, ds = None, fetch_models=True):
    clfs = []
    if args.dataset == 'CENSUS':
        original_targets = ["0.0", "0.1", "0.2", "0.3","0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
        targets = sorted(targets)
        if use_original_targets:
            targets_to_use = original_targets
        else:
            targets_to_use = targets + new_targets

        params_verify = []
        labels_verify=[]
        label_vals_verify = []
        clfs_to_ret = []
        # Load the final test data
        num_test_models = 100
        for index, verify_ratio in enumerate(targets_to_use):
            print("Ratio: ", verify_ratio)
            if verify_ratio in new_targets:
                X_verify, verify_labels, _, clfs = meta_utils.get_model_representations(meta_utils.get_models_path(result_path,"prover/test", args.filter + "", verify_ratio), args, 1, args.first_n, verify_model_performance = args.verify_base, fetch_models=fetch_models)
                # 100 models
                clfs_to_ret += clfs
            else:
                print("Other: ", verify_ratio)
                X_verify, verify_labels, _, clfs = meta_utils.get_model_representations(meta_utils.get_models_path(result_path,"prover/test", args.filter + "", verify_ratio), args, 0, args.first_n, verify_model_performance=args.verify_base, fetch_models=fetch_models)
                # if len(targets) == 8:
                #     X_verify, verify_labels = X_verify[:38], verify_labels[:38]
                # else:
                #     X_verify, verify_labels = X_verify[:22], verify_labels[:22]
                num_test_to_take = (num_test_models * len(new_targets)) // len(targets)
                print(f'num_test_to_take: {num_test_to_take}')
                X_verify, verify_labels = X_verify[:num_test_to_take], verify_labels[:num_test_to_take]
                clfs_to_ret += clfs[:num_test_to_take]
            if return_target_ratio_vals:
                verify_label_values = [float(verify_ratio)]*len(verify_labels)
                label_vals_verify += verify_label_values
            params_verify.append(X_verify)
            labels_verify.append(verify_labels)

        params_verify = np.concatenate(params_verify)
        params_verify = meta_utils.prepare_batched_data(params_verify)
        X_verify = params_verify # Why is this needed?
        labels_verify = np.concatenate(labels_verify)
        labels_verify = torch.from_numpy(labels_verify)
        if return_target_ratio_vals:
            label_vals_verify = np.array(label_vals_verify, dtype=np.float32)
            label_vals_verify = torch.from_numpy(label_vals_verify)
            return params_verify, labels_verify, label_vals_verify, clfs_to_ret
        return params_verify, labels_verify, clfs_to_ret
    
    elif args.dataset == 'BONEAGE':
        verify_neg_w_list = []
        verify_pos_w_list = []
        label_vals_verify = []
        label_vals_verify_pos = []
        label_vals_verify_neg = []
        clfs_to_ret_neg = []
        clfs_to_ret_pos = []
        for index, tg in enumerate(["0.2", "0.3","0.4", "0.5", "0.6", "0.7", "0.8"]):
            verify_dir = os.path.join(result_path, "prover/test/%s" % tg)
            _, vecs_verify, clfs = meta_utils.get_model_features(verify_dir, args, first_n=args.first_n,start_n=args.start_n, prune_ratio=args.prune_ratio,max_read=1000, verify_model_performance = args.verify_base, fetch_models=fetch_models)
            if tg in new_targets:
                print("Pos Ratio", tg)
                verify_pos_w_list += vecs_verify
                if return_target_ratio_vals:
                    verify_label_values = [float(tg)]*len(vecs_verify)
                    label_vals_verify_pos += verify_label_values
                clfs_to_ret_pos += clfs
            elif tg in targets:
                print("Neg Ratio", tg)
                num_to_choose = 75
                if (args.claim_ratio == "0.2") or (args.claim_ratio == "0.8"):
                    num_to_choose = 60
                verify_neg_w_list += vecs_verify[:num_to_choose]

                if return_target_ratio_vals:
                    verify_label_values = [float(tg)]*len(vecs_verify[:num_to_choose])
                    label_vals_verify_neg += verify_label_values

                clfs_to_ret_neg += clfs[:num_to_choose]

        vecs_verify_neg = np.array(verify_neg_w_list, dtype='object')
        vecs_verify_pos = np.array(verify_pos_w_list, dtype='object')
        X_verify = np.concatenate((vecs_verify_neg, vecs_verify_pos))
        label_vals_verify = label_vals_verify_neg + label_vals_verify_pos
        clfs_to_ret = clfs_to_ret_neg + clfs_to_ret_pos
        Y_verify = [0.] * len(vecs_verify_neg) + [1.] * len(vecs_verify_pos)
        Y_verify = torch.from_numpy(np.array(Y_verify)).to(device)

        if return_target_ratio_vals:
            label_vals_verify = np.array(label_vals_verify, dtype=np.float32)
            label_vals_verify = torch.from_numpy(label_vals_verify)
            return X_verify, Y_verify, label_vals_verify, clfs_to_ret

        return X_verify, Y_verify, clfs_to_ret

    elif args.dataset == 'ARXIV':
        X_verify_pos, X_verify_neg = [], []
        Y_verify_pos, Y_verify_neg = [], []
        verify_dirs_pos = [os.path.join(result_path, "prover/test",  x) for x in new_targets]
        verify_dirs_neg = [os.path.join(result_path, "prover/test",  x) for x in targets]
        label_vals_verify = []
        clfs_to_ret = []
        for i, (trd, ratio) in enumerate(zip(verify_dirs_pos, new_targets)):
            print("Pos Ratio:", ratio)
            dims, vecs_train, clfs = models.get_model_features_arxiv(trd, ds, args, max_read=args.train_sample + args.val_sample, verify_model=args.verify_base, fetch_models=fetch_models)
            X_verify_pos += vecs_train
            Y_verify_pos.append([1] * len(vecs_train))
            if return_target_ratio_vals:
                verify_label_values = [float(ratio)]*len(vecs_train)
                label_vals_verify += verify_label_values
            clfs_to_ret += clfs

        for i, (trd,ratio) in enumerate(zip(verify_dirs_neg,targets)):
            print("Neg Ratio: {}".format(ratio))
            dims, vecs_train, clfs = models.get_model_features_arxiv(trd, ds, args, max_read=args.train_sample + args.val_sample, verify_model=args.verify_base, fetch_models=fetch_models)
            X_verify_neg += vecs_train[:100]
            Y_verify_neg.append([0] * len(vecs_train[:100]))
            if return_target_ratio_vals:
                verify_label_values = [float(ratio)]*len(vecs_train[:100])
                label_vals_verify += verify_label_values
            clfs_to_ret += clfs[:100]

        Y_verify_pos = torch.from_numpy(np.concatenate(Y_verify_pos)).float().to(device)
        Y_verify_neg = torch.from_numpy(np.concatenate(Y_verify_neg)).float().to(device)
        Y_verify = torch.cat((Y_verify_pos,Y_verify_neg))

        X_verify_pos = np.array(X_verify_pos, dtype='object')
        X_verify_neg = np.array(X_verify_neg, dtype='object')
        X_verify = np.concatenate((X_verify_pos,X_verify_neg))
        X_verify = meta_utils.prepare_batched_data(X_verify)

        if return_target_ratio_vals:
            label_vals_verify = np.array(label_vals_verify, dtype=np.float32)
            label_vals_verify = torch.from_numpy(label_vals_verify)
            return X_verify, Y_verify, label_vals_verify, clfs_to_ret
        return X_verify, Y_verify, clfs_to_ret