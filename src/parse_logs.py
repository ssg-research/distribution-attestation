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

# This script is to parse the results logs from experiments

import matplotlib.pyplot as plt
import argparse
import logging
import os
import csv
import numpy as np
import pandas as pd
from .hybrid_fpr_far import make_hybrid_plots, normalize

def parse_and_collect(args, log):
    attestation_classifier_best_accs = []
    threshs = []
    acc_aucrocs = []
    rej_aucrocs = []
    threshold_acc_lb = []
    threshold_acc_ub = []
    threshold_acc_mean = []
    f1s = []
    precs = []
    recalls = []
    fpr_neg = []
    fpr_pos = []
    fnr_neg = []
    fnr_pos = []
    fprs = []
    fnrs = []
    tprs = []
    tnrs = []
    tar_at_1 = []
    tar_at_5 = []
    tar_at_10 = []
    trr_at_1 = []
    trr_at_5 = []
    trr_at_10 = []

    # The base model stats cannot be zipped with the attestation classifier stats
    basemodel_mean_accs = []
    basemodel_stddev_accs = []

    lb_idx = 2
    ub_idx = 3
    mean_dx = -1
    if args.filter != '' and args.filter[0] != '_':
        args.filter = '_'+args.filter[0]
    logfilename = os.path.join('src', args.logfile_subdir, args.dataset+args.filter+"_"+args.claim_property+"_"+args.logfile_suffix+".log")
    with open(logfilename, 'r') as f:
        for line in f:
            line_parts = line.strip().split(':')
            sentence_parts = line_parts[0].split(' ')
            if line_parts[0] == 'Best Representation Generator Accuracy':
                attestation_classifier_best_accs.append(float(line_parts[-1].strip()))
                print(attestation_classifier_best_accs[-1])
                print()
            # Base model stats will come up next
            elif sentence_parts[0] == 'Mean' and sentence_parts[1] == 'accuracy':
                basemodel_mean_accs.append(float(line_parts[-1].strip()))
            elif sentence_parts[0] == 'Standard' and sentence_parts[1] == 'deviation':
                basemodel_stddev_accs.append(float(line_parts[-1].strip()))
            elif line_parts[0] == '[acceptance AUCROC] Best Threshold':
                threshs.append(float(line_parts[1].split(';')[0].strip()))
                acc_aucrocs.append(float(line_parts[-1].strip()))
                print(threshs[-1])
                print(acc_aucrocs[-1])
                print()
            elif line_parts[0] == '[rejection AUCROC] Best Threshold':
                threshs.append(float(line_parts[1].split(';')[0].strip()))
                rej_aucrocs.append(float(line_parts[-1].strip()))
                print(threshs[-1])
                print(rej_aucrocs[-1])
                print()
            elif line_parts[0] == 'Accuracy':
                threshold_acc_lb.append(float(line_parts[lb_idx].split(";")[0].strip()))
                threshold_acc_ub.append(float(line_parts[ub_idx].split(";")[0].strip()))
                threshold_acc_mean.append(float(line_parts[mean_dx].strip()))
                print(threshold_acc_lb[-1])
                print(threshold_acc_ub[-1])
                print(threshold_acc_mean[-1])
                print()
            elif line_parts[0] == 'F1 Score':
                f1s.append(float(line_parts[-1].strip()))
                print(f1s[-1])
            elif line_parts[0] == 'Precision Score':
                precs.append(float(line_parts[-1].strip()))
                print(precs[-1])
            elif line_parts[0] == 'Recall Score':
                recalls.append(float(line_parts[-1].strip()))
                print(recalls[-1])
                print()
            elif line_parts[0] == 'False Acceptance Rate':
                fprs.append(float(line_parts[-1].strip()))
                print(fprs[-1])
                print()
            elif line_parts[0] == 'False Rejection Rate':
                fnrs.append(float(line_parts[-1].strip()))
                print(fnrs[-1])
                print()
            elif line_parts[0] == 'True Acceptance Rate':
                tprs.append(float(line_parts[-1].strip()))
                print(tprs[-1])
                print()
            elif line_parts[0] == 'True Rejection Rate':
                tnrs.append(float(line_parts[-1].strip()))
                print(tnrs[-1])
                print()
            elif line_parts[0] == 'TAR at FAR 1%':
                tar_at_1.append(float(line_parts[-1].strip()))
                print(tar_at_1[-1])
                print()
            elif line_parts[0] == 'TAR at FAR 5%':
                tar_at_5.append(float(line_parts[-1].strip()))
                print(tar_at_5[-1])
                print()
            elif line_parts[0] == 'TAR at FAR 10%':
                tar_at_10.append(float(line_parts[-1].strip()))
                print(tar_at_10[-1])
                print()
            elif line_parts[0] == 'TRR at FRR 1%':
                trr_at_1.append(float(line_parts[-1].strip()))
                print(trr_at_1[-1])
                print()
            elif line_parts[0] == 'TRR at FRR 5%':
                trr_at_5.append(float(line_parts[-1].strip()))
                print(trr_at_5[-1])
                print()
            elif line_parts[0] == 'TRR at FRR 10%':
                trr_at_10.append(float(line_parts[-1].strip()))
                print(trr_at_10[-1])
                print()

    print(attestation_classifier_best_accs)
    print(threshs)
    print(acc_aucrocs)
    print(rej_aucrocs)
    print(threshold_acc_lb)
    print(threshold_acc_ub)
    print(threshold_acc_mean)
    print(f1s)
    print(recalls)
    print(fprs)
    print(fnrs)
    print(tprs)
    print(tnrs)

    precision = 1 / ((2/f1s[-1]) - 1/recalls[-1])

    print(basemodel_mean_accs)
    print(basemodel_stddev_accs)

    if len(tprs) > 0 and len(tnrs) > 0:
        return [round(acc_aucrocs[-1],2)], [round(rej_aucrocs[-1],2)], [round(threshold_acc_mean[-1],2)], \
            [round(precision, 2)], [round(recalls[-1], 2)]
    
    return [round(acc_aucrocs[-1],2)], [round(rej_aucrocs[-1],2)], [round(threshold_acc_mean[-1],2)], \
            [round(precision, 2)], [round(recalls[-1], 2)]

            
def save_tables(args, results, add_ratio = False, at_low_ratio = False):

    if add_ratio:
        main_fields = ["Ratio", "Acceptance AUCROC", "Rejection AUCROC", "Mean Accuracy (%)","Precision", "Recall", "FAR", "FRR", "TAR", "TRR"]
    else:
        main_fields = ["Acceptance AUCROC", "Rejection AUCROC", "Mean Accuracy (%)", "Precision", "Recall", "FAR", "FRR", "TAR", "TRR"]
    
    if at_low_ratio:
        main_fields = main_fields[:-4]
        main_fields.append('TAR @ FAR 1%')
        main_fields.append('TAR @ FAR 5%')
        main_fields.append('TAR @ FAR 10%')
        main_fields.append('TRR @ FNR 1%')
        main_fields.append('TRR @ FNR 5%')
        main_fields.append('TRR @ FNR 10%')

    print(results)
    print()
    zipped_results = np.array(results)#.T
    print(zipped_results)
    dataset_str = args.dataset + args.dataset_suffix
    base_path = os.path.join(args.plots_path, dataset_str)

    if add_ratio:
        file_name = os.path.join(base_path, "full_table_"+args.filter)
    else:
        file_name = os.path.join(base_path, args.claim_property)

    with open(file_name+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(main_fields)
        writer.writerows(zipped_results)

    
    print("CSV file saved.")



def gather_finetuning_results(args):
    original_accs = []
    prefinetuned_accs = []
    postfinetuned_accs = []
    if args.filter != '' and args.filter[0] != '_':
        args.filter = '_'+args.filter[0]
    logfilename = os.path.join(args.logfile_subdir, args.dataset+args.filter+"_"+"bb"+"_"+args.claim_property+args.logfile_suffix+".log")
    take_original_acc_next_iter = False
    take_prefine_acc_net_iter = False
    with open(logfilename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'Base model performance prior to weight replacement':
                take_original_acc_next_iter = True
                continue
            elif line == 'Base model performance following weight replacement':
                take_prefine_acc_net_iter = True
            

            line_parts = line.split(':')
            if line_parts[0].strip() == 'Accuracy acheived':
                if take_original_acc_next_iter:
                    original_accs.append(float(line_parts[-1].strip()))
                    take_original_acc_next_iter = False
                elif take_prefine_acc_net_iter:
                    prefinetuned_accs.append(float(line_parts[-1].strip()))
                    take_prefine_acc_net_iter = False
                continue

            if args.dataset != 'ARXIV':
                line_parts = line.split(' ')
                if line_parts[0].strip() == 'New' and line_parts[1].strip() == 'accuracy' \
                    and line_parts[2].strip() == 'for' and line_parts[3].strip() == 'model[0]':
                    postfinetuned_accs.append(float(line_parts[5].strip()))
            else:
                line_parts = line.split(';')
                pertinent_part = line_parts[-1]
                pertinent_part_split = pertinent_part.split(':')
                if pertinent_part_split[0].strip() == 'Test accuracy new':
                    postfinetuned_accs.append(float(pertinent_part_split[-1].strip()))

    print('Finished collecting accuracies')
    print(original_accs[-1])
    print(prefinetuned_accs[-1])
    print(postfinetuned_accs[-1])
    print()

    mean_original_acc = np.mean(original_accs)
    mean_prefinetuned_acc = np.mean(prefinetuned_accs)
    mean_postfinetuned_acc = np.mean(postfinetuned_accs)

    std_original_acc = np.std(original_accs)
    std_prefinetuned_acc = np.std(prefinetuned_accs)
    std_postfinetuned_acc = np.std(postfinetuned_accs)

    print(f'Mean original accuracy: {mean_original_acc}')
    print(f'Mean prefinetuned accuracy: {mean_prefinetuned_acc}')
    print(f'Mean postfinetuned accuracy: {mean_postfinetuned_acc}')
    print()
    print(f'std original accuracy: {std_original_acc}')
    print(f'std prefinetuned accuracy: {std_prefinetuned_acc}')
    print(f'std postfinetuned accuracy: {std_postfinetuned_acc}')
    print()

    # return original_accs, prefinetuned_accs, postfinetuned_accs

def plot_tar_or_trr(args, tars_or_trrs, far_frr_vals, online_inf_cost, number_of_accepts, num_actual_negative, false_accepts, total_provers, specific_metric = 'FAR'):
    # property_vals = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    # if args.dataset == 'ARXIV':
    #     property_vals = ['9', '10', '11', '12', '13', '14', '15', '16', '17']
    # if args.dataset == 'BONEAGE':
    #     property_vals = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
    
    args.specific_metric = specific_metric
    args.online_cost_inf = online_inf_cost[-1]

    for tar_or_trr_val, far_or_frr_val in zip(tars_or_trrs, far_frr_vals):
        args.tar_inference = tar_or_trr_val 
        args.far_inference = far_or_frr_val
        args.number_of_accepts = number_of_accepts[-1]
        args.num_actual_negative = num_actual_negative[-1]
        args.num_false_accepts = false_accepts[-1]
        args.normalization_maximum = total_provers
        make_hybrid_plots(args)


def hybrid_analysis(args):
    print(f'Now parsing {args.dataset} with property {args.claim_property}')
    parse_verify = not args.parse_test
    tar_at_1 = []
    tar_at_5 = []
    tar_at_10 = []
    trr_at_1 = []
    trr_at_5 = []
    trr_at_10 = []

    true_acc_at_far5 = []
    false_acc_at_far5 = []
    true_rej_at_far5 = []
    false_rej_at_far5 = []

    true_acc_at_frr5 = []
    false_acc_at_frr5 = []
    true_rej_at_frr5 = []
    false_rej_at_frr5 = []

    prec_at_far5 = []
    prec_at_frr5 = []

    online_inf_cost = []
    number_of_accepts_frr5 = []
    number_of_rejects_frr5 = []
    num_actual_negatives_frr5 = []
    num_actual_positives_frr5 = []

    number_of_accepts_far5 = []
    number_of_rejects_far5 = []
    num_actual_negatives_far5 = []
    num_actual_positives_far5 = []
    if args.filter != '' and args.filter[0] != '_':
        args.filter = '_'+args.filter[0]
    logfilename = os.path.join('src', args.logfile_subdir, args.dataset+args.filter+"_bb_"+args.claim_property+"_"+args.logfile_suffix+".log")
    print_verification_seen = False
    performance_on_clean_set_seen = False
    with open(logfilename, 'r') as f:
        for line in f:
            # Only start logging after we see the log message "Printing verification next"
            print(line)
            if line.strip() == 'Printing verification next':
                print_verification_seen = True
                print('verify seen')
                if parse_verify:
                    continue
            elif line.strip() == 'Performance on clean set':
                performance_on_clean_set_seen  = True
                if parse_verify:
                    continue
            if parse_verify:
                if not print_verification_seen and not performance_on_clean_set_seen:
                    continue
            else:
                if print_verification_seen and performance_on_clean_set_seen:
                    break                   
            # print('Both seen')
            line_parts = line.strip().split(':')
            # if len(online_inf_cost) > 0:
            # if line_parts[0] == 'TAR at FAR 1%':
            #     tar_at_1.append(float(line_parts[-1].strip()))
            #     print(f'TAR at FAR 1%: {tar_at_1[-1]}')
            #     print()
            if line_parts[0] == 'TAR at FAR 5%':
                tar_at_5.append(float(line_parts[-1].strip()))
                print(f'TAR at FAR 5%: {tar_at_5[-1]}')
                print()
            elif line_parts[0] == 'TRR at FRR 5%':
                trr_at_5.append(float(line_parts[-1].strip()))
                print(f'TRR at FRR 5%: {trr_at_5[-1]}')
                print()
            elif line_parts[0] == 'True accepts at FAR 5%':
                true_acc_at_far5.append(float(line_parts[-1].strip()))
                print(f'True accepts at FAR 5%: {true_acc_at_far5}')
                print()
            elif line_parts[0] == 'False accepts at FAR 5%':
                false_acc_at_far5.append(float(line_parts[-1].strip()))
                print(f'False accepts at FAR 5%: {false_acc_at_far5}')
                print()
            elif line_parts[0] == 'True rejects at FAR 5%':
                true_rej_at_far5.append(float(line_parts[-1].strip()))
                print(f'True rejects at FAR 5%: {true_rej_at_far5}')
                print()
            elif line_parts[0] == 'False rejects at FAR 5%':
                false_rej_at_far5.append(float(line_parts[-1].strip()))
                print(f'False rejects at FAR 5%: {false_rej_at_far5}')
                print()

            
            elif line_parts[0] == 'True accepts at FRR 5%':
                true_acc_at_frr5.append(float(line_parts[-1].strip()))
                print(f'True accepts at FRR 5%: {true_acc_at_frr5}')
                print()
            elif line_parts[0] == 'False accepts at FRR 5%':
                false_acc_at_frr5.append(float(line_parts[-1].strip()))
                print(f'False accepts at FRR 5%: {false_acc_at_frr5}')
                print()
            elif line_parts[0] == 'True rejects at FRR 5%':
                true_rej_at_frr5.append(float(line_parts[-1].strip()))
                print(f'True rejects at FRR 5%: {true_rej_at_frr5}')
                print()
            elif line_parts[0] == 'False rejects at FRR 5%':
                false_rej_at_frr5.append(float(line_parts[-1].strip()))
                print(f'False rejects at FRR 5%: {false_rej_at_frr5}')
                print()
                if parse_verify:
                    break

            elif line_parts[0] == 'Precision Score at FAR 5%':
                prec_at_far5.append(float(line_parts[-1].strip()))
            elif line_parts[0] == 'Precision Score at FRR 5%':
                prec_at_frr5.append(float(line_parts[-1].strip()))

            elif line_parts[0] == 'Total inference time taken' and parse_verify:
                online_inf_cost.append(float(line_parts[-1].strip()))
            elif line_parts[0] == 'Total inference time taken on test' and not parse_verify:
                online_inf_cost.append(float(line_parts[-1].strip()))
                break
    
    for ta, fa in zip(true_acc_at_frr5, false_acc_at_frr5):
        number_of_accepts_frr5.append(ta + fa)
        print(f'number of accepts at FRR 5%: {ta+fa}')
    
    for tr, fa in zip(true_rej_at_frr5, false_acc_at_frr5):
        num_actual_negatives_frr5.append(tr+fa)
        print(f'num actual negatives at FRR 5%: {tr+fa}')
    
    for ta, fr in zip(true_acc_at_frr5, false_rej_at_frr5):
        num_actual_positives_frr5.append(ta+fr)
        print(f'num actual positives at FRR 5%: {ta+fr}')
    
    for tr, fr in zip(true_rej_at_frr5, false_rej_at_frr5):
        number_of_rejects_frr5.append(tr+fr)
        print(f'number of rejects at FRR 5%: {tr+fr}')
    


    for ta, fa in zip(true_acc_at_far5, false_acc_at_far5):
        number_of_accepts_far5.append(ta+fa)
        print(f'number of accepts at FAR 5%: {ta+fa}')
    
    for tr, fa in zip(true_rej_at_far5, false_acc_at_far5):
        num_actual_negatives_far5.append(tr+fa)
        print(f'num actual negatives at FAR 5%: {tr+fa}')
    
    for ta, fr in zip(true_acc_at_far5, false_rej_at_far5):
        num_actual_positives_far5.append(ta+fr)
        print(f'num actual positives at FAR 5%: {ta+fr}')
    
    for tr, fr in zip(true_rej_at_far5, false_rej_at_far5):
        number_of_rejects_far5.append(tr+fr)
        print(f'number of rejects at FAR 5%: {tr+fr}')

    
    # We have the total time taken to go through all the provers.
    # The following is to get the cost per prover
    total_provers = tr+ta+fr+fa
    online_inf_cost[-1] = online_inf_cost[-1]/total_provers
    
    tars = [0.95] # FRR is 0.05
    fars = [1-np.array(trr_at_5)] # we care about the case where the frr is fixed (and trr is poor, hence high far)
    # print(f'far at frr 5%: {fars[-1]}')
    far_at_frr5_precise = [false_acc_at_frr5[-1] / (false_acc_at_frr5[-1] + true_rej_at_frr5[-1])]
    
    # Print the normalized versions of each of the above
    args.normalization_maximum = total_provers
    print(f'Total provers: {total_provers}')
    print(f'Normalized number_of_accepts_at_frr5: {normalize(args, number_of_accepts_frr5[-1])}')
    print(f'Normalized num actual positives at frr5: {normalize(args, num_actual_positives_frr5[-1])}')
    print(f'Normalized num actual negatives at frr5: {normalize(args, num_actual_negatives_frr5[-1])}')
    print(f'Normalized number of rejects at frr5: {normalize(args, number_of_rejects_frr5[-1])}')
    print(f'Normalized number of false accepts ar frr5: {normalize(args, false_acc_at_frr5[-1])}')
    print()
    print(f'Normalized number_of_accepts_at_far5: {normalize(args, number_of_accepts_far5[-1])}')
    print(f'Normalized num actual positives at far5: {normalize(args, num_actual_positives_far5[-1])}')
    print(f'Normalized num actual negatives at far5: {normalize(args, num_actual_negatives_far5[-1])}')
    print(f'Normalized number of rejects at far5: {normalize(args, number_of_rejects_far5[-1])}')
    print(f'Normalized number of false rejects ar far5: {normalize(args, false_rej_at_far5[-1])}')
    print()
    
    plot_tar_or_trr(args, tars, far_at_frr5_precise, online_inf_cost, number_of_accepts_frr5, num_actual_negatives_frr5, false_acc_at_frr5, total_provers, specific_metric='FAR')


def get_files_with_prefix(directory, prefix):
    file_list = []

    # Iterate over the files in the directory
    dir_list = os.listdir(directory)
    dir_list.sort()
    for filename in dir_list:
        if filename.startswith(prefix):
            file_list.append(filename)

    return file_list


def gather_hybrid_results(args):
    # Get all of the files corresponding to the dataset in question
    dir_to_search = os.path.join('src', args.logfile_subdir)
    logfiles = get_files_with_prefix(dir_to_search, args.dataset)
    relevant_lines = []
    costs = []
    fars = []
    for logfile in logfiles:
        if args.dataset == 'CENSUS':
            # Do not process the files that do not have _r_ in it if the filter is race
            if args.filter == 'race':
                if logfile.split('_r_')[0] == logfile:
                    continue
            elif args.filter == 'sex':
                if logfile.split('_s_')[0] == logfile:
                    continue
        
        print(logfile)
        logfilename = os.path.join(dir_to_search, logfile)
        with open(logfilename, 'r') as f:
            for line in f:
                line_split_parts = line.split(' ')
                if line_split_parts[0] == 'FAR' or line_split_parts[0] == 'Cost':
                    print(line)
                    relevant_lines.append(line)
                    if line_split_parts[0] == 'FAR':
                        fars.append(float(line.split(':')[-1].strip()))
                    else:
                        costs.append(line.split(':')[-1].strip())
        print()
    return costs, fars


def plot_bars(args, costs, fars, plot_cost_on_y = False):
    property_vals = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    if args.dataset == 'ARXIV':
        property_vals = ['9', '10', '11', '12', '13', '14', '15', '16', '17']
    if args.dataset == 'BONEAGE':
        property_vals = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
    
    bar_width = 0.2  # Width of each bar
    index = np.arange(len(property_vals))  # X coordinates for the bars

    fars_inf_only = fars[0::3]
    fars_mid = fars[1::3]
    fars_crypto = fars[2::3]
    costs_inf_only = costs[0::3]
    costs_mid = costs[1::3]
    costs_crypto = costs[2::3]

    # The crypto cost is after the plus and before the C (last character)
    costs_inf_only = [round(float(txt.split('+')[-1][:-1]), 2) for txt in costs_inf_only]
    costs_mid = [round(float(txt.split('+')[-1][:-1]), 2) for txt in costs_mid]
    costs_crypto = [round(float(txt.split('+')[-1][:-1]), 2) for txt in costs_crypto]

    print(f'MID COSTS: {costs_mid}')

    text1 = costs_inf_only
    text2 = costs_mid
    text3 = costs_crypto
    y1 = fars_inf_only
    y2 = fars_mid
    y3 = fars_crypto

    if plot_cost_on_y:

        fars_inf_only = [round(num,2) for num in fars_inf_only]
        fars_mid = [round(num,2) for num in fars_mid]
        fars_crypto = [round(num,2) for num in fars_crypto]

        y1 = costs_inf_only
        y2 = costs_mid
        y3 = costs_crypto
        text1 = fars_inf_only
        text2 = fars_mid
        text3 = fars_crypto

    plt.bar(index, y1, width=bar_width, label='num_samples = 0 (Inference only)')
    plt.bar(index + bar_width, y2, width=bar_width, label='num_samples = FA at FRR 5%')
    plt.bar(index + 2 * bar_width, y3, width=bar_width, label='num_samples = N_acc (Full crypto)')

    plt.xlabel('Degree')
    if plot_cost_on_y:
        plt.ylabel('Crypto Cost (number of provers)')
    else:
        plt.ylabel('FAR')
    plt.xticks(index + bar_width, property_vals)
    plt.legend()

    # Add text labels on top of the bars
    for i in range(len(property_vals)):
        plt.text(index[i], y1[i], text1[i], ha='center', va='bottom')
        plt.text(index[i] + bar_width, y2[i], text2[i], ha='center', va='bottom')
        plt.text(index[i] + 2*bar_width, y3[i], text3[i], ha='center', va='bottom')

    plt.show()
    plt.savefig(f'{args.dataset}_{args.filter}_{args.claim_property}_bars.png')



def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--plots_path', type=str, default='plots/', help="Directory to store the plots")
    parser.add_argument('--dataset', type=str, default="ARXIV", help='Options: CENSUS, BONEAGE')
    parser.add_argument('--dataset_suffix', type=str, default='', help='CENSUS, BONEAGE, or ARXIV could have some suffix at the end for their plots directory')                     
    parser.add_argument('--claim_property', default="122")
    parser.add_argument('--logfile_subdir', default='', type=str, help='Sometimes the logifle may be in a subdirectory within src/')
    parser.add_argument('--logfile_suffix', default="test", help = 'The logfile may have some text before .log, that text is represented in')
    parser.add_argument('--filter', type=str,default="",required=False,help='which filter to use')
    parser.add_argument('--finetuned_performance_check', action='store_true', default=False, help='Gather the basemodel performance results before and after finetuning')
    parser.add_argument('--hybrid_analysis', action='store_true', default=False, help='Calculate the hybrid FAR/FRR/cost results from parsing the logs')
    parser.add_argument('--plot_pmfs', action='store_true', default=False, help='Plot the pmfs. This option may be used during hybrid analysis')
    parser.add_argument('--gather_hybrid_results', action = 'store_true', default=False, help = 'THIS OPTION IS ONLY ALLOWED AFTER COMPLETING HYBRID ANALYSIS. Go through each  of the logs and get the results')
    parser.add_argument('--parse_test', action='store_true', default=False, help='Use this option if you wish to extract true/false accept/reject from test data instead of verify data (Default)')
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="train_verifier_model_exp.log", filemode="w")
    log: logging.Logger = logging.getLogger("PropertyAttestation")
    args = handle_args()
    assert(not (args.hybrid_analysis and args.gather_hybrid_results))
    if args.finetuned_performance_check:
        gather_finetuning_results(args)
    elif args.hybrid_analysis:
        hybrid_analysis(args)
    elif args.gather_hybrid_results:
        costs, fars = gather_hybrid_results(args)
        plot_bars(args, costs, fars)
    else:
        results = parse_and_collect(args, log)
        save_tables(args, results)