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

import numpy as np
import torch
import os
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List
import pprint
import statistics
from . import models
from . import data
from . import utils
import matplotlib.pyplot as plt
import scipy.stats
from numpy import argmax
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from ogb.nodeproppred import Evaluator
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_meta_model(args, model, train_data, test_data,epochs, lr, eval_every=5, binary=True, regression=False, val_data=None, batch_size=1000, combined=False):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    loss_fn = nn.BCEWithLogitsLoss()
    params, y = train_data
    params_test, y_test = test_data
    y = y.to(device)
    y_test = y_test.to(device)

    iterator = tqdm(range(epochs))
    best_acc = 0
    best_model = None
    for e in iterator:
        model.train()

        rp_tr = np.random.permutation(y.shape[0])
        if not combined:
            assert('Not combined no longer supported')
            # params, y = params[rp_tr], y[rp_tr]
        else:
            y = y[rp_tr]
            params = [x[rp_tr] for x in params]

        running_acc, loss, num_samples = 0, 0, 0
        i = 0

        if combined:
            n_samples = len(params[0])
        else:
            n_samples = len(params)

        while i < n_samples:
            outputs = []
            if not combined:
                assert('Not combined no longer supported')
                # for param in params[i:i+batch_size]:
                #     param = [a.to(device) for a in param]
                #     outputs.append(model(param)[:, 0])

            else:
                param_batch = [x[i:i+batch_size] for x in params]
                param_batch = [a.to(device) for a in param_batch]

                outputs.append(model(param_batch)[:, 0])

            outputs = torch.cat(outputs, 0)

            optimizer.zero_grad()
            loss = loss_fn(outputs, y[i:i+batch_size])
            loss.backward()
            optimizer.step()
            num_samples += outputs.shape[0]
            loss += loss.item() * outputs.shape[0]

            print_acc = ""
            running_acc += acc_fn(outputs, y[i:i+batch_size], binary)
            print_acc = ", Accuracy: %.2f" % (float(100 * running_acc) / float(num_samples))

            iterator.set_description("Epoch %d : [Train] Loss: %.5f%s" % (e+1, loss / num_samples, print_acc))
            i += batch_size

        # Evaluate on test data now
        if (e+1) % eval_every == 0:

            t_acc, t_loss = test_meta(args,model, loss_fn, params_test, y_test, batch_size, acc_fn, combined=combined)
            
            if best_acc < t_acc:
                best_model = model
                best_acc = t_acc

            print_acc = ""
            print_acc = ", Test Accuracy: %.2f" % (t_acc)
            iterator.set_description("[Test] Loss: %.5f%s" % (t_loss, print_acc))

    model.eval()

    if args.dataset == 'ARXIV':
        print("Best Model Accuracy: {}".format(best_acc))
        return best_model, best_acc
    return model, t_acc

@torch.no_grad()
def test_meta(args, model, loss_fn, X, Y, batch_size, accuracy, combined=False, return_output = False):
    model.eval()
    loss, num_samples, running_acc = 0, 0, 0
    i = 0
    if combined:
        n_samples = len(X[0])
    else:
        n_samples = len(X)

    while i < n_samples:
        outputs = []
        if not combined:
            assert('Not combined no longer supported')

        else:
            param_batch = [x[i:i+batch_size] for x in X]
            param_batch = [a.to(device) for a in param_batch]
            outputs.append(model(param_batch)[:, 0])

        outputs = torch.cat(outputs, 0)

        num_samples += outputs.shape[0]

        loss += loss_fn(outputs, Y[i:i+batch_size]).item() * num_samples

        running_acc += accuracy(outputs, Y[i:i+batch_size]).item()
        i += batch_size

    if return_output:
        return outputs, 100 * running_acc / num_samples, loss / num_samples
    return 100 * running_acc / num_samples, loss / num_samples

def acc_fn(x, y, binary=True):
    if binary:
        return torch.sum((y == (x >= 0)))
    return torch.sum(y == torch.argmax(x, 1))


def get_representation_generator_output(args, rep_gen, X, Y, combined = False):
    loss_fn = nn.BCEWithLogitsLoss()
    batch_size=1000
    return get_meta_results_for_thresholding(X, rep_gen, batch_size, combined = combined)


def plots(args,return_embeddings_nonclaim, nonclaim_labels, plots_path = Path('plots/'), claim_labels = None, return_embeddings_claim = None):

    tsne_transform = TSNE(n_components=2, perplexity=10)
    X_embedded = tsne_transform.fit_transform(return_embeddings_nonclaim)

    nonclaim_embedding_list = []
    for i in range(len(set(nonclaim_labels))):
        nonclaim_embedding_list.append(X_embedded[nonclaim_labels==i])

    colors = ['red','blue','green','black','slateblue','orange','lightseagreen','slategray','m','pink','brown']
    fig = plt.figure(figsize=(4.5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    for index, lst in enumerate(nonclaim_embedding_list):
        ax.scatter(lst[:, 0], lst[:, 1], color=colors[index], ls='solid', marker='.',label='%s'%str(index),alpha=0.6)
    plt.tick_params(left=False, bottom = False)
    plt.legend(loc="best",fontsize="large")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    if args.dataset == "CENSUS":
        plt.savefig("nonclaim_embeddings_{}_claim_{}_filter_{}.pdf".format(args.dataset,args.claim_ratio,args.filter), bbox_inches = 'tight',pad_inches = 0.01, dpi=400)
    else:
        plt.savefig("nonclaim_embeddings_{}_claim_{}.pdf".format(args.dataset,args.claim_ratio), bbox_inches = 'tight',pad_inches = 0.01, dpi=400)
    print('Saved plots')
    plt.close()

def perform_ensemble(models, param, threshes, label_vals_verify = None, verbose=False, better_rep_gen = ''):
    results = [models[0](param)[:, 0], models[1](param)[:, 0]]
    dists = [(results[0] - threshes[0]).detach().cpu().numpy().tolist(), (results[1] - threshes[1]).detach().cpu().numpy().tolist()]

    better_model_idx = 1
    if better_rep_gen == 'lower':
        better_model_idx = 0
    other_model_idx = -1*(better_model_idx - 1)

    res = []
    # It seems that regardless of the model, they are more sure about ratios that are further away from target
    for i, dist in enumerate(dists[better_model_idx]):
        if dist < 0:
            res.append(0)
        else:
            if dists[other_model_idx][i] < 0:
                res.append(0)
            else:
                res.append(1)

    return res


def get_meta_results_for_thresholding(x_data, model, batch_size, combined = False, threshes = [], plot_tsne = False, label_vals_verify = None, better_rep_gen = ''):
    outputs = []
    i = 0
    if plot_tsne:
        x_reformat = []

    if not combined:
        raise Exception('Not Combined is no longer being allowed to happen- only Combined accepted')
    else:
        param_batch = x_data[i:i+batch_size]
        # param_batch = param_batch.to(device) # This line is if param_batch is not set to be a list. Robustness expects non-lists
        param_batch = [a.to(device) for a in param_batch]
        if plot_tsne:
            # param_batch_cpu = param_batch.to('cpu')
            param_batch_cpu = [a.to('cpu') for a in param_batch]
            x_reformat.append(param_batch_cpu)
        if isinstance(model, list) and len(model) >= 2:
            res = perform_ensemble(model, param_batch, threshes, label_vals_verify = label_vals_verify, better_rep_gen = better_rep_gen)
            final_result = torch.tensor(res)
        else:
            # We send all the data (batch contains all the data)
            final_result = model(param_batch)
            final_result = final_result[:, 0]
        outputs.append(final_result)
    
    if plot_tsne:
        return outputs, x_reformat
    return outputs


def analyze_predictions_by_ratio(outputs_verify, y_verify, label_vals_verify = None, make_plot = False):
    if label_vals_verify is None:
        return
    
    paired = list(zip(outputs_verify, y_verify, label_vals_verify))
    label_vals = {}
    for pred, ground_truth, label in paired:
        if round(label.item(),2) not in label_vals:
            label_vals[round(label.item(),2)] = {'Correct': 0, 'Incorrect':0}
        if int(pred) == ground_truth:
            label_vals[round(label.item(),2)]['Correct'] += 1
        else:
            label_vals[round(label.item(),2)]['Incorrect'] += 1

    if make_plot:
        fig = plt.figure(figsize=(4.5, 3.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(label_vals_verify.numpy(), outputs_verify)
        plt.savefig('repgen_distrib_vals.png')

    
    pprint.pprint(label_vals)


def test_meta_threshold(args, plots_path, model, x_test, y_test, x_verify, y_verify, batch_size, threshes = [], combined=True, existence=False, plot_tsne = False, \
                        return_thresh = False, label_vals_verify = None, do_not_plot=False, better_rep_gen = '', return_minmax = False):
    if isinstance(model, list) and len(model) >= 2:
        model[0] = model[0].eval()
        model[1] = model[1].eval()
    else:
        model.eval()
    loss, running_acc = 0, 0
    i = 0
    # if combined:
    #     n_samples_test = len(x_test[0])
    #     n_samples_verify = len(x_verify[0])
    # else:
    # Combined will no longer be needed
    n_samples_test = len(x_test)
    n_samples_verify = len(x_verify)

    aucroc_threshold = None
    if model.aucroc_thresh is not None:
        print(f'Model has its own aucroc: {model.aucroc_thresh}')
        aucroc_threshold = model.aucroc_thresh
        model.aucroc_thresh = None

    start_test = time.time()
    outputs_test = get_meta_results_for_thresholding(x_test, model, batch_size, combined, threshes, label_vals_verify = None, better_rep_gen=better_rep_gen)
    outputs_test = torch.cat(outputs_test, 0)
    end_test = time.time()

    start = time.time()
    outputs_verify, x_reformat = get_meta_results_for_thresholding(x_verify, model, batch_size, combined, threshes, plot_tsne=True, label_vals_verify=label_vals_verify, better_rep_gen=better_rep_gen)
    outputs_verify = torch.cat(outputs_verify, 0)
    end = time.time()

    print('Analyzing predictions before thresholding')
    analyze_predictions_by_ratio(outputs_verify.detach().cpu(), y_verify, label_vals_verify, make_plot=True)
    print(f'Going to potentially compute threshold now')

    fpr, tpr, fnr, tnr = None, None, None, None
    min_test = np.min(outputs_test.cpu().detach().numpy())
    max_test = np.max(outputs_test.cpu().detach().numpy())
    min_verify = np.min(outputs_verify.cpu().detach().numpy())
    max_verify = np.max(outputs_verify.cpu().detach().numpy())
    if len(threshes) == 0:
        outputs_test_for_thresh = (outputs_test.cpu().detach().numpy() - min_test) / (max_test - min_test)
        outputs_verify_for_thresh = (outputs_verify.cpu().detach().numpy() - min_verify) / (max_verify - min_verify)

        threshold_new, fpr, tpr, fnr, tnr = models.compute_threshold(y_test[i:i+batch_size].cpu().detach().numpy(),outputs_test_for_thresh,args,plots_path, existence, do_not_plot=do_not_plot)#, print_ta_fa_at_5 = False)
        print(f'Total inference time taken on test: {end_test - start_test}')
        print('Printing verification next')
        print(f'Total inference time taken: {end-start}')
        models.compute_threshold(y_verify[i:i+batch_size].cpu().detach().numpy(),outputs_verify_for_thresh,args,plots_path, existence, do_not_plot=do_not_plot)
        if aucroc_threshold is None:
            outputs_verify_for_thresh = (outputs_verify.cpu().detach().numpy() - np.min(outputs_verify.cpu().detach().numpy())) / (np.max(outputs_verify.cpu().detach().numpy()) - np.min(outputs_verify.cpu().detach().numpy()))
            aucroc_threshold = threshold_new
            y_pred_test = outputs_test_for_thresh > aucroc_threshold
            y_pred_verify = outputs_verify_for_thresh > aucroc_threshold
        else:
            y_pred_test = outputs_test > aucroc_threshold
            y_pred_verify = outputs_verify > aucroc_threshold
        
    else:
        y_pred_test = outputs_test
        y_pred_verify = outputs_verify
        if aucroc_threshold is None:
            aucroc_threshold, fpr, tpr, fnr, tnr = models.compute_threshold(y_test[i:i+batch_size].cpu().detach().numpy(),outputs_test.detach().cpu().numpy(),args,plots_path, existence, do_not_plot=do_not_plot)

    analyze_predictions_by_ratio(y_pred_verify, y_verify, label_vals_verify)

    if plot_tsne:
        new_x = []
        for arr in x_reformat[0][0]:
            new_x.append(arr)
        x_reformat = np.stack(new_x)
        plots(args, x_reformat, y_pred_verify)

    if return_thresh:
        if return_minmax:
            return y_pred_test, y_test.cpu(), y_pred_verify, y_verify.cpu(), aucroc_threshold, [fpr, tpr, fnr, tnr], [min_test, max_test]
        return y_pred_test, y_test.cpu(), y_pred_verify, y_verify.cpu(), aucroc_threshold, [fpr, tpr, fnr, tnr]
    if return_minmax:
        return y_pred_test, y_test.cpu(), y_pred_verify, y_verify.cpu(), [fpr,tpr,fnr,tnr], [min_test, max_test]
    return y_pred_test, y_test.cpu(), y_pred_verify, y_verify.cpu(), [fpr,tpr,fnr,tnr]


def compute_metrics(y_pred_test, y_test, y_pred_verify, y_verify, args = None, plots_path = None):
    acc_test = balanced_accuracy_score(y_verify,y_pred_verify)
    confidence = 0.95
    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((acc_test * (1 - acc_test)) / y_pred_verify.shape[0])
    ci_lower = acc_test - ci_length
    ci_upper = acc_test + ci_length
    if ci_upper > 1:
        ci_upper = 1
    conf_matrix = confusion_matrix(y_verify, y_pred_verify)
    true_positives = np.diag(conf_matrix)
    false_positives = conf_matrix.sum(axis=0) - true_positives
    false_negatives = conf_matrix.sum(axis=1) - true_positives
    true_negatives = conf_matrix.sum() - (false_positives + false_negatives + true_positives)
    fpr, tpr, thresholds = roc_curve(y_verify, y_pred_verify)
    roc_auc_verify = roc_auc_score(y_verify, y_pred_verify)
    print("Accuracy: Lower Bound: {:.2f}; Upper Bound: {:.2f}; Mean: {:.2f}".format(round(ci_lower*100,2), round(ci_upper*100,2), round(acc_test*100,2)))
    print(f'ROC AUC score on verify: {roc_auc_verify}')
    print("F1 Score: {:.2f}".format(round(f1_score(y_verify,y_pred_verify),2)))
    print("Precision Score: {:.2f}".format(round(precision_score(y_verify,y_pred_verify),2)))
    print("Recall Score: {:.2f}".format(round(recall_score(y_verify,y_pred_verify),2)))
    print(f"False Acceptance Rate: {(false_positives/(false_positives+true_negatives))[-1]}")
    # print(f"False Positive Rate: {fpr}")
    print(f"False Rejection Rate: {(false_negatives/(false_negatives+true_positives))[-1]}")
    print(f'True Acceptance Rate: {(true_positives/(true_positives+false_negatives))[-1]}')
    print(f'True Rejection Rate: {(true_negatives/(true_negatives+false_positives))[-1]}')

    if args is not None and plots_path is not None:
        print(f'IF THE FOLLOWING "TAR/TRR AT FAR/FRR" WERE OUTPUTTED PREVIOUSLY, PLEASE USE THOSE VALUES AND DISREGARD')
        print(f'THE FOLLOWING "TAR/TRR AT FAR/FRR" VALUES ARE USED IN SPECIAL CASES DURING ADVERSARIAL ATTACK/DEFENSE')
        tpr_at_fpr001 = tpr[np.where(fpr.round(2)<=0.01)]
        tpr_at_fpr005 = tpr[np.where(fpr.round(2)<=0.05)]
        tpr_at_fpr01 = tpr[np.where(fpr.round(2)<=0.1)]
        # print(f'tpr_at_fpr_01: {tpr_at_fpr01}')
        val_tpr_at_fpr001 = round(tpr_at_fpr001.tolist()[-1], 2)
        val_tpr_at_fpr005 = round(tpr_at_fpr005.tolist()[-1], 2)
        val_tpr_at_fpr01 = round(tpr_at_fpr01.tolist()[-1], 2)
        print(f'TAR at FAR 1%: {val_tpr_at_fpr001}')
        print(f'TAR at FAR 5%: {val_tpr_at_fpr005}')
        print(f'TAR at FAR 10%: {val_tpr_at_fpr01}')

        # Flip the labels to find the true negative vs false negative
        inv_labels = (y_verify - 1)*-1
        inv_outputs = (y_pred_verify - 1)*-1
        fnr, tnr, thresholds = roc_curve(inv_labels, inv_outputs)
        tnr_at_fnr001 = tnr[np.where(fnr.round(2)<=0.01)]
        tnr_at_fnr005 = tnr[np.where(fnr.round(2)<=0.05)]
        tnr_at_fnr01 = tnr[np.where(fnr.round(2)<=0.1)]
        val_tnr_at_fnr001 = round(tnr_at_fnr001.tolist()[-1], 2)
        val_tnr_at_fnr005 = round(tnr_at_fnr005.tolist()[-1], 2)
        val_tnr_at_fnr01 = round(tnr_at_fnr01.tolist()[-1], 2)
        print(f'TRR at FRR 1%: {val_tnr_at_fnr001}')
        print(f'TRR at FRR 5%: {val_tnr_at_fnr005}')
        print(f'TRR at FRR 10%: {val_tnr_at_fnr01}')

        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        print(f'eer_threshold: {eer_threshold}')
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(f'EER from fpr: {EER}')
        EER_verify = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(f'EER sanity check from fnr: {EER_verify}')

        aucroc_threshold = None
        return round(acc_test*100,2), thresholds, aucroc_threshold, fpr, tpr, fnr, tnr
    
    return round(acc_test*100,2)




def get_trained_meta_features(model, train_data, test_data, batch_size=1000, combined=False):

    params_train, y_train = train_data
    params_test, y_test = test_data
    y_train, y_test = y_train.to(device), y_test.to(device)

    model.eval()

    rp_tr = np.random.permutation(y_train.shape[0])
    rp_te = np.random.permutation(y_test.shape[0])
    if not combined:
        assert('Not combined no longer supported')
        # params_train, y = params_train[rp_tr], y_train[rp_tr]
        # params_test, y_test = params_test[rp_te], y_test[rp_te]
    else:
        y_train, y_test = y_train[rp_tr], y_test[rp_te]
        params_train, params_test = [x[rp_tr] for x in params_train], [x[rp_te] for x in params_test]
    i=0
    rep_train = []
    if not combined:
        assert('Not combined no longer supported')
        # for param_train in params_train[i:i+batch_size]:
        #     param_train = [a.to(device) for a in param_train]
        #     rep_train.append(model(param_train,getrep=True).cpu().detach()[0].numpy())
        # rep_train = np.array(rep_train)

    else:
        param_batch_train = [x[i:i+batch_size] for x in params_train]
        param_batch_train = [a.to(device) for a in param_batch_train]
        rep_train = model(param_batch_train,getrep=True).cpu().detach().numpy()
        rep_train = np.array(rep_train)

    rep_test = []
    if not combined:
        assert('Not combined no longer supported')
        # for param_test in params_test[i:i+batch_size]:
        #     param_test = [a.to(device) for a in param_test]
        #     rep_test.append(model(param_test,getrep=True).cpu().detach()[0].numpy())
        # rep_test = np.array(rep_test)

    else:
        param_batch_test = [x[i:i+batch_size] for x in params_test]
        param_batch_test = [a.to(device) for a in param_batch_test]
        rep_test = model(param_batch_test,getrep=True).cpu().detach().numpy()
        rep_test = np.array(rep_test)
    return rep_train, y_train.numpy(), rep_test, y_test.numpy()

def get_trained_meta_features_test(model, params_test, batch_size=1000, combined=False):

    model.eval()

    rp_te = np.random.permutation(params_test.shape[0])
    if not combined:
        assert('Not combined no longer supported')
        # params_test = params_test[rp_te]
    else:
        params_test = [x[rp_te] for x in params_test]
    i=0

    rep_test = []
    if not combined:
        assert('Not combined no longer supported')
        # for param_test in params_test[i:i+batch_size]:
        #     param_test = [a.to(device) for a in param_test]
        #     rep_test.append(model(param_test,getrep=True).cpu().detach()[0].numpy())
        # rep_test = np.array(rep_test)

    else:
        param_batch_test = [x[i:i+batch_size] for x in params_test]
        param_batch_test = [a.to(device) for a in param_batch_test]
        rep_test = model(param_batch_test,getrep=True).cpu().detach().numpy()
        rep_test = np.array(rep_test)
    return rep_test


def verify_basemodel_performance(args, clfs, model_dir, ds = None, finetune = False, weights = None, biases = None, return_preds = False):
    # NOTE: This function currently does not support overlap between prover and verifier datasets
    if finetune and args.dataset == 'CENSUS':
        assert(weights is not None and biases is not None)
    # Load the data
    raw_path = Path("dataset/")
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    all_accs = []
    ratio = model_dir.split('/')[-1]
    preds_on_data = []
    # print(f'ratio is: {ratio}')

    if args.dataset == "CENSUS":
        raw_path = raw_path / "census_splits"
        dataset = data.CensusWrapper(path=raw_path,filter_prop=args.filter, ratio=float(ratio), split="prover")
        (x_tr, y_tr), (x_te, y_te), cols = dataset.load_data()
        # Send to test
        for i, clf in enumerate(clfs):
            avg_acc = 100 * clf.score(x_te, y_te.ravel())
            all_accs.append(avg_acc)

            if return_preds:
                preds_on_data.append(clf.predict(x_te))

            # If we are finetuning, train the model some more and return the finetuned model
            if finetune:
                for j in range(clf.n_iter_):
                    clf.partial_fit(x_tr, y_tr.ravel())
                    clf.coefs_[0] = weights
                    clf.intercepts_[0] = biases
                new_acc = 100 * clf.score(x_te, y_te.ravel())
                print(f'New accuracy for model[{i}] is {new_acc} where old acc was {avg_acc}')
                clfs[i] = clf

                # Verify that the first layer weights are still the ones we inserted
                # NOTE: this is fine for now because we only have the one classifier that we are testing
                if (clf.coefs_[0] == weights).all():
                    print('Weights correct')
                else:
                    print('Weights incorrect')
                if (clf.intercepts_[0] == biases).all():
                    print('Biases correct')
                else:
                    print('Biases incorrect')


    elif args.dataset == 'BONEAGE':
        # assert(args.dataset == 'BONEAGE')
        raw_path = raw_path / "boneage_splits"

        def filter1(x): return x["gender"] == 1

        df_train, df_val = data.get_df(raw_path, "prover")
        features = data.get_features(raw_path, "prover")

        # n_train, n_test = 700, 200
        # if args.split == "prover":
        n_train, n_test = 1400, 400

        df_train_processed = data.heuristic(df_train, filter1, float(ratio), n_train, class_imbalance=1.0, n_tries=300)
        df_val_processed = data.heuristic(df_val, filter1, float(ratio), n_test, class_imbalance=1.0, n_tries=300)
        ds = data.BoneWrapper(df_train_processed,df_val_processed,features=features,augment=False)
        train_loader, val_loader = ds.get_loaders(256*32, shuffle=True)

        # Send to test
        for i, clf in enumerate(clfs):
            if return_preds:
                avg_loss, avg_acc, preds_received = utils.validate_epoch(val_loader, clf.to(device), criterion, verbose = False, return_preds=True)
                preds_on_data.append(preds_received)
            else:
                avg_loss, avg_acc = utils.validate_epoch(val_loader, clf.to(device), criterion, verbose = False)

            all_accs.append(avg_acc)

            if finetune:
                _, new_acc = utils.train(clf, (train_loader, val_loader),lr=1e-3,epoch_num=100,weight_decay=1e-4)
                print(f'New accuracy for model[{i}] is {new_acc} where old acc was {avg_acc}')
                clfs[i] = clf

                # Verify that the first layer weights are still the ones we inserted
                state_dict = clf.state_dict()
                if (state_dict['layers.0.weight'] == weights.to(device)).all():
                    print('Weights correct')
                else:
                    print('Weights incorrect')
                if (state_dict['layers.0.bias'] == biases.to(device)).all():
                    print('Biases correct')
                else:
                    print('Biases incorrect')
    
    elif args.dataset == 'ARXIV':
        assert args.prune >= 0 and args.prune <= 0.1 # Prune ratio should be valid and not too large
        if ds is None:
            ds = data.ArxivNodeDataset("verifier")
            # Need to save this one

            # n_edges = ds.g.number_of_edges()
            # n_nodes = ds.g.number_of_nodes()
            # print("""----Data statistics------'\n #Edges %d; #Nodes %d; #Classes %d""" %(n_edges, n_nodes, ds.num_classes,))

            # print("-> Before modification")
            # print("Nodes: %d, Average degree: %.2f" %(ds.num_nodes, ds.g.number_of_edges() / ds.num_nodes))
            # print("Train: %d, Test: %d" % (len(ds.train_idx), len(ds.test_idx)))

            # ds.change_mean_degree(int(args.claim_degree), args.prune)

            # print("-> After modification")
            # print("Nodes: %d, Average degree: %.2f" %(ds.num_nodes, ds.g.number_of_edges() / ds.num_nodes))
            # print("Train: %d, Test: %d" % (len(ds.train_idx), len(ds.test_idx)))
            # pick_tr, pick_te = 62000, 35000
            # ds.label_ratio_preserving_pick(pick_tr, pick_te)

        evaluator = Evaluator(name='ogbn-arxiv')

        for i, clf in enumerate(clfs):
            train_idx, test_idx = ds.get_idx_split()
            if return_preds:
                train_acc, test_acc, preds_received = models.test_arxiv(clf.cpu(), ds, train_idx, test_idx, evaluator, return_preds = True)
                preds_on_data.append(preds_received)
            else:
                train_acc, test_acc = models.test_arxiv(clf.cpu(), ds, train_idx, test_idx, evaluator)

            all_accs.append(test_acc)

            if finetune:
                run_accs = models.train_model_arxiv(ds, clf, evaluator, args)
                acc_tr, acc_te = run_accs["train"][-1], run_accs["test"][-1]
                print("Train accuracy old: {:.2f}; Test accuracy old: {:.2f}".format(train_acc*100,test_acc*100))
                print("Train accuracy new: {:.2f}; Test accuracy new: {:.2f}".format(acc_tr*100,acc_te*100))
                clfs[i] = clf
            
                # Verify that the first layer weights are still the ones we inserted
                state_dict = clf.state_dict()
                if (state_dict['layers.0.weight'].cuda() == weights.to(device)).all():
                    print('Weights correct')
                else:
                    print('Weights incorrect')
                if (state_dict['layers.0.bias'].cuda() == biases.to(device)).all():
                    print('Biases correct')
                else:
                    print('Biases incorrect')


    if len(all_accs) > 1:
        mean_acc = statistics.mean(all_accs)
        std_dev_acc = statistics.stdev(all_accs)

        print(f'Mean accuracy of loaded {args.dataset} model is: {mean_acc}')
        print(f'Standard deviation of accuracy for loaded {args.dataset} model is: {std_dev_acc}')
    else:
        if len(all_accs) > 0:
            print(f'Accuracy acheived: {all_accs[0]}')

    if args.dataset == 'ARXIV':
        if finetune:
            return ds, clfs
        if return_preds:
            return ds, preds_on_data
        return ds
    if finetune:
        return clfs
    if return_preds:
        return preds_on_data
    return None
            


########## CENSUS ##########

def flash_utils(args):
    print("==> Arguments:")
    for arg in vars(args):
        print(arg, " : ", getattr(args, arg))

def prepare_batched_data(X):
    inputs = [[] for _ in range(len(X[0]))]
    for x in X:
        for i, l in enumerate(x):
            inputs[i].append(l)
    inputs = np.array([torch.stack(x, 0) for x in inputs], dtype='object')
    return inputs


def get_model_representations(folder_path, args, label, first_n=np.inf, n_models=1000, start_n=0, fetch_models: bool = False, shuffle: bool = True, models_provided: bool = False, verify_model_performance: bool = False):
    """
        If models_provided is True, folder_path will actually be a list of models.
    """
    if models_provided:
        models_in_folder = folder_path
    else:
        models_in_folder = os.listdir(folder_path)

    if shuffle:
        np.random.shuffle(models_in_folder)

    models_in_folder = models_in_folder[:n_models]
    w, labels, clfs = [], [], []
    for path in tqdm(models_in_folder):
        if models_provided:
            clf = path
        else:
            clf = models.load_model_census(os.path.join(folder_path, path))
        if fetch_models or verify_model_performance:
            clfs.append(clf)

        weights = [torch.from_numpy(x) for x in clf.coefs_]
        dims = [w.shape[0] for w in weights]
        biases = [torch.from_numpy(x) for x in clf.intercepts_]
        processed = [torch.cat((w, torch.unsqueeze(b, 0)), 0).float().T for (w, b) in zip(weights, biases)]

        if first_n != np.inf:
            processed = processed[start_n:first_n]
            dims = dims[start_n:first_n]

        w.append(processed)
        labels.append(label)

    labels = np.array(labels)

    w = np.array(w, dtype=object)
    labels = torch.from_numpy(labels)

    if verify_model_performance:
        verify_basemodel_performance(args, clfs, folder_path)

    return w, labels, dims, clfs


def get_models_path(results_path,property, split, value=None):
    if value is None:
        return os.path.join(results_path, property, split)
    return os.path.join(results_path,  property, split, value)



######## BONEAGE ########
import torch.nn.utils.prune as prune


def get_weight_layers(m, normalize=False, transpose=True,first_n=np.inf, start_n=0,custom_layers=None,conv=False, include_all=False,prune_mask=[]):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0
    custom_layers = sorted(custom_layers) if custom_layers is not None else None

    track = 0
    for name, param in m.named_parameters():
        if "weight" in name:
            param_data = param.data.detach().cpu()
            if len(prune_mask) > 0:
                param_data = param_data * prune_mask[track]
                track += 1

            if transpose:
                param_data = param_data.T

            weights.append(param_data)
            if conv:
                dims.append(weights[-1].shape[2])
                dim_kernels.append(weights[-1].shape[0] * weights[-1].shape[1])
            else:
                dims.append(weights[-1].shape[0])
        if "bias" in name:
            biases.append(torch.unsqueeze(param.data.detach().cpu(), 0))

        i += 1

        if custom_layers is None:
            if (i - 1) // 2 < start_n:
                dims, dim_kernels, weights, biases = [], [], [], []
                continue
            if i // 2 > first_n - 1:
                break
        else:
            if i // 2 != custom_layers[j // 2]:
                dims = dims[:-1]
                dim_kernels = dim_kernels[:-1]
                weights = weights[:-1]
                biases = biases[:-1]
            else:
                j += 1

            if len(custom_layers) == j // 2:
                break

    if custom_layers is not None and len(custom_layers) != j // 2:
        raise ValueError("Custom layers requested do not match actual model")

    if include_all:
        if conv:
            middle_dim = weights[-1].shape[3]
        else:
            middle_dim = weights[-1].shape[1]

    if normalize:
        min_w = min([torch.min(x).item() for x in weights])
        max_w = max([torch.max(x).item() for x in weights])
        weights = [(w - min_w) / (max_w - min_w) for w in weights]
        weights = [w / max_w for w in weights]

    cctd = []
    for w, b in zip(weights, biases):
        if conv:
            b_exp = b.unsqueeze(0).unsqueeze(0)
            b_exp = b_exp.expand(w.shape[0], w.shape[1], 1, -1)
            combined = torch.cat((w, b_exp), 2).transpose(2, 3)
            combined = combined.view(-1, combined.shape[2], combined.shape[3])
        else:
            combined = torch.cat((w, b), 0).T

        cctd.append(combined)

    if conv:
        if include_all:
            return (dims, dim_kernels, middle_dim), cctd
        return (dims, dim_kernels), cctd
    if include_all:
        return (dims, middle_dim), cctd
    return dims, cctd


def get_model_features(model_dir, args, max_read=None, first_n=np.inf,start_n=0, prune_ratio=None,shift_to_gpu: bool = True,fetch_models: bool = False,models_provided: bool = False,shuffle: bool = False, verify_model_performance: bool = False):
    vecs, clfs = [], []

    if models_provided:
        iterator = model_dir
    else:
        iterator = os.listdir(model_dir)

    if shuffle:
        np.random.shuffle(iterator)

    encountered = 0

    for mpath in tqdm(iterator):
        if models_provided:
            model = mpath
        else:
            if os.path.isdir(os.path.join(model_dir, mpath)):
                continue

            model = models.load_model_boneage(os.path.join(model_dir, mpath), cpu=not shift_to_gpu)

        if fetch_models or verify_model_performance:
            clfs.append(model)

        prune_mask = []
        if prune_ratio is not None:
            for layer in model.layers:
                if type(layer) == nn.Linear:
                    prune.l1_unstructured(layer, name='weight', amount=prune_ratio)
                    prune_mask.append(layer.weight_mask.data.detach().cpu())


        dims, fvec = get_weight_layers(model, first_n=first_n, start_n=start_n,prune_mask=prune_mask)
        fvec = [x.to('cpu') for x in fvec]
        vecs.append(fvec)
        encountered += 1
        if encountered == max_read:
            break

    if verify_model_performance:
        verify_basemodel_performance(args, clfs, model_dir)

    return dims, vecs, clfs





#################### ARXIV #######################
# Arxiv code combined with above

