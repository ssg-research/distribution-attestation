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

from venv import create
from sklearn.neural_network import MLPClassifier
import os
from pathlib import Path
from joblib import load, dump
import torch.nn as nn
import torch
from ogb.nodeproppred import Evaluator
import statistics
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from numpy import argmax
from . import data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_curve(args, fr, tr, labels, outputs, thresholds, plots_path, existence = False, plot_rejection = False, save_fig=True, fig = None, ax = None, do_not_plot=False):
    marker = 'o'
    ls = 'solid'
    label = 'FRR-TRR'
    color='blue'

    if fig is None and ax is None:
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='medium')
        plt.rc('ytick', labelsize='medium')
        plt.rc('grid', linestyle="dotted", color='black')

        fig = plt.figure(figsize=(4.5, 3.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0,1], [0,1], linestyle='--', color='black', label='Random')

        if plot_rejection:
            ax.set_xlabel('False Rejection Rate', fontsize='large')
            ax.set_ylabel('True Rejection Rate', fontsize='large')
            prefix = "rejection"
        else:
            ax.set_xlabel('False Acceptance Rate', fontsize='large')
            ax.set_ylabel('True Acceptance Rate', fontsize='large')    
            prefix = "acceptance"
            marker = 'x'
            label = 'FAR-TAR'
            ls = 'dotted'
            color='black'

    else:
        ax.set_xlabel('False Rejection/Acceptance Rate', fontsize='large')
        ax.set_ylabel('True Rejection/Acceptance Rate', fontsize='large')
        prefix = 'rejection'
        # Always false in 'both' case
        if not plot_rejection:
            label = 'FAR-TAR'
            marker = 'x'
            ls = 'dotted'
            color='black'

    J = tr - fr
    ix = argmax(J)

    aucroc_threshold = thresholds[ix]
    ax.plot(fr, tr, color=color, ls=ls, marker='.', markersize=2, label=label+' Curve')
    # ax.scatter(fr[ix], tr[ix], marker=marker, color=color, label=f'Best {prefix} thresh')
    ax.legend()

    print('[{} AUCROC] Best Threshold: {:.2f}; {} AUCROC: {:.2f}'.format(prefix, thresholds[ix], prefix, roc_auc_score(labels, outputs)))
    if not do_not_plot:
        if save_fig:
            if existence:
                if args.dataset == "CENSUS":
                    plt.savefig(os.path.join(plots_path,prefix+"_existence_aucroc_{}_{}.pdf".format(args.dataset,args.filter)), bbox_inches = 'tight',pad_inches = 0, dpi=400)
                else:
                    plt.savefig(os.path.join(plots_path,prefix+"_existence_aucroc_{}.pdf".format(args.dataset)), bbox_inches = 'tight',pad_inches = 0, dpi=400)
            else:
                if args.dataset == "CENSUS":
                    plt.savefig(os.path.join(plots_path,prefix+"_attestation_aucroc_{}_{}_{}.pdf".format(args.dataset,args.filter,args.claim_ratio)), bbox_inches = 'tight',pad_inches = 0, dpi=400)
                else:
                    if args.dataset == 'BONEAGE':
                        suffix = args.claim_ratio
                    else:
                        suffix = args.claim_degree
                    plt.savefig(os.path.join(plots_path,prefix+"_attestation_aucroc_{}_{}.pdf".format(args.dataset,suffix)), bbox_inches = 'tight',pad_inches = 0, dpi=400)
                    print(f'Saved to {os.path.join(plots_path,prefix+"_attestation_aucroc_{}_{}.pdf".format(args.dataset,suffix))}')
            return aucroc_threshold
    return aucroc_threshold, fig, ax


def compute_threshold(labels,outputs,args,plots_path, existence = False, do_not_plot = False, print_ta_fa_at_5 = True):
    # Binary labels, and the lengths match, as expected
    # print(f'labels: {labels}')
    # print(f'labels length: {len(labels)}')
    # print(f'outputs shape: {outputs.shape}')

    fpr, tpr, thresholds = roc_curve(labels, outputs)
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

    thresh_at_fpr005 = thresholds[np.where(fpr.round(2)<=0.05)]
    val_thresh_at_fpr005 = round(thresh_at_fpr005.tolist()[-1], 2)
    rethresh_outputs_fpr005 = outputs > val_thresh_at_fpr005

    conf_matrix_fpr005 = confusion_matrix(labels, rethresh_outputs_fpr005)
    true_positives_fpr005 = np.diag(conf_matrix_fpr005)
    false_positives_fpr005 = conf_matrix_fpr005.sum(axis=0) - true_positives_fpr005
    false_negatives_fpr005 = conf_matrix_fpr005.sum(axis=1) - true_positives_fpr005
    true_negatives_fpr005 = conf_matrix_fpr005.sum() - (false_positives_fpr005 + false_negatives_fpr005 + true_positives_fpr005)
    if print_ta_fa_at_5:
        print("F1 Score at FAR 5%: {:.2f}".format(round(f1_score(labels,rethresh_outputs_fpr005),2)))
        print("Precision Score at FAR 5%: {:.2f}".format(round(precision_score(labels,rethresh_outputs_fpr005),2)))
        print("Recall Score at FAR 5%: {:.2f}".format(round(recall_score(labels,rethresh_outputs_fpr005),2)))
        print(f"True accepts at FAR 5%: {true_positives_fpr005[-1]}")
        print(f"True rejects at FAR 5%: {true_negatives_fpr005[-1]}")
        print(f"False accepts at FAR 5%: {false_positives_fpr005[-1]}")
        print(f"False rejects at FAR 5%: {false_negatives_fpr005[-1]}")

    aucroc_threshold, fig, ax = plot_curve(args, fpr, tpr, labels, outputs, thresholds, plots_path, existence=existence, save_fig=False, do_not_plot=do_not_plot)

    # Flip the labels to find the true negative vs false negative
    inv_labels = (labels - 1)*-1
    inv_outputs = (outputs - 1)*-1
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

    thresh_at_fnr005 = thresholds[np.where(fnr.round(2)<=0.05)]
    val_thresh_at_fnr005 = round(thresh_at_fnr005.tolist()[-1], 2)
    rethresh_outputs_fnr005 = inv_outputs > val_thresh_at_fnr005

    conf_matrix_fnr005 = confusion_matrix(inv_labels, rethresh_outputs_fnr005)
    true_negatives_fnr005 = np.diag(conf_matrix_fnr005)
    false_negatives_fnr005 = conf_matrix_fnr005.sum(axis=0) - true_negatives_fnr005
    false_positives_fnr005 = conf_matrix_fnr005.sum(axis=1) - true_negatives_fnr005
    true_positives_fnr005 = conf_matrix_fnr005.sum() - (false_positives_fnr005 + false_negatives_fnr005 + true_negatives_fnr005)
    if print_ta_fa_at_5:
        print("F1 Score at FRR 5%: {:.2f}".format(round(f1_score(inv_labels,rethresh_outputs_fnr005),2)))
        print("Precision Score at FRR 5%: {:.2f}".format(round(precision_score(inv_labels,rethresh_outputs_fnr005),2)))
        print("Recall Score at FRR 5%: {:.2f}".format(round(recall_score(inv_labels,rethresh_outputs_fnr005),2)))
        print(f"True accepts at FRR 5%: {true_positives_fnr005[-1]}")
        print(f"True rejects at FRR 5%: {true_negatives_fnr005[-1]}")
        print(f"False accepts at FRR 5%: {false_positives_fnr005[-1]}")
        print(f"False rejects at FRR 5%: {false_negatives_fnr005[-1]}")

    try:
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        print(f'eer_threshold: {eer_threshold}')
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(f'EER from fpr: {EER}')
        EER_verify = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(f'EER sanity check from fnr: {EER_verify}')

        # print(fnr)
        # print(tnr)
        plot_curve(args, fnr, tnr, inv_labels, inv_outputs, thresholds, plots_path, plot_rejection = True, existence=existence, fig=fig, ax=ax, do_not_plot=do_not_plot)
    except:
        print('Could not calculate EER or plot_curve')

    return aucroc_threshold, fpr, tpr, fnr, tnr


class GenerateRepresentation2(torch.nn.Module):
    def __init__(self, dims, inside_dims=[64, 8], n_classes=2, dropout = 0.5, aucroc_thresh = None, use_logits = False):
        super(GenerateRepresentation2, self).__init__()
        self.dims = dims
        self.dropout = dropout
        self.aucroc_thresh = aucroc_thresh
        self.use_logits = use_logits
        self.layers = []
        prev_layer = 0

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        def make_mini(y):
            layers = [torch.nn.Linear(y, inside_dims[0]),torch.nn.ReLU()]
            for i in range(1, len(inside_dims)):
                layers.append(torch.nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(self.dropout))
            return torch.nn.Sequential(*layers)

        # for census, dims is just [42]
        for i, dim in enumerate(self.dims):
            # +1 for bias
            # prev_layer for previous layer
            # input dimension per neuron
            if i > 0:
                prev_layer = inside_dims[-1] * dim
            self.layers.append(make_mini(prev_layer + 1 + dim))

        self.layers = torch.nn.ModuleList(self.layers)
        
        # Final network to combine them all together
        self.rho = nn.Linear(inside_dims[-1] * len(dims), n_classes)

    def forward(self, param, getrep=False, do_inference = False):
        # reps = []
        prev_layer_reps = None
        # print(f'len(params): {len(params)}') # 1 for Boneage, Arxiv, Census-R, Census-S
        if isinstance(param, list):
            is_batched = len(param[0].shape) > 2
            param = param[0]
            # print(f'param is list')
        else:
            is_batched = len(param.shape) > 2
        # print(f'Is batched?: {is_batched}') # Boneage used to not be batched, the other two are - make them all batched now (ie. a list of 1 tensor (batched))

        # 600 x 1 for Boneage, each is a list, with 128 x 1025 in it. (600 lists)
        # Arxiv is a list with a 1600 x 256 x 129 in it
        # Census is a list with a 998 x 32 x 43

        # For labels
        # 600 for Boneage
        # 1600 for Arxiv
        # 998 for Census

        # 3D
        # Case where data is batched per layer (batch_size x _ x _)
        # print(f'param shape: {param.shape}')
        # print(f'Number of layers: {len(self.layers)}')
        # print(f'Number of layers[0]: {len(self.layers[0])}')
        # print(f'dims: {self.dims}')
        if is_batched:
            param_eff = param.view(-1, param.shape[-1])
            prev_shape = param.shape
            '''print(f'param_eff viewed: {param_eff.view(-1, param_eff.shape[-1]).shape}')'''
            processed = self.layers[0](param_eff)
            processed_final = processed.view(prev_shape[0], prev_shape[-2], -1) #prev_shape[0] because batch_size comes first
        # 2D - unused (Boneage used to use this)
        else:
            param_eff = param
            processed_final = self.layers[0](param_eff)

        # Store this layer's representation
        # reps.append(torch.sum(processed_final, -2))
        reps_c = torch.sum(processed_final, -2)
        # There is only one layer (first layer)

        # if is_batched:
        #     reps_c = torch.cat(reps, 1)
        # else:
        #     reps_c = torch.unsqueeze(torch.cat(reps), 0)

        if getrep:
            return reps_c
        '''print(f'reps_c shape: {reps_c.shape}')'''
        logits = self.rho(reps_c)
        '''print(f'logits shape: {logits.shape}')'''
        '''print()'''
        if self.aucroc_thresh is not None:
            # print(f'logits: {logits}')
            # print(f'aucroc thresh: {self.aucroc_thresh}')

            # y_pred = (logits >= self.aucroc_thresh).to(dtype=torch.float32).retain_grad()
            # (if logits > aucroc, predict 1, else 0)
            diffs = self.aucroc_thresh - logits
            if self.use_logits:
                diffs_second_logits = logits - self.aucroc_thresh
                y_pred = torch.cat((diffs, diffs_second_logits), dim=-1)
                return y_pred

            # print(diffs)
            y_pred = torch.sigmoid(diffs)
            # print(y_pred)

            # # Make into a softmax style 2-class prediction
            y_pred_inv = (y_pred-1.)*-1.
            # ones = torch.ones_like(y_pred_inv)
            # print((ones==y_pred_inv).all())
            y_pred = torch.cat((y_pred, y_pred_inv), dim=-1)
            # print(y_pred)

            # print(f'y_pred shape: {y_pred.shape}')
            # print()
            # print(y_pred[0])
            return y_pred
        # aucroc_threshold, fpr, tpr, fnr, tnr = compute_threshold(y_test[i:i+batch_size].cpu().detach().numpy(),logits,args,plots_path, existence, do_not_plot=do_not_plot)
        # y_pred_test = outputs_test > aucroc_threshold
        return logits

class GenerateRepresentation(torch.nn.Module):
    def __init__(self, dims, inside_dims=[64, 8], n_classes=2, dropout = 0.5):
        super(GenerateRepresentation, self).__init__()
        self.dims = dims
        self.dropout = dropout
        self.layers = []
        prev_layer = 0

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        def make_mini(y):
            layers = [torch.nn.Linear(y, inside_dims[0]),torch.nn.ReLU()]
            for i in range(1, len(inside_dims)):
                layers.append(torch.nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(self.dropout))
            return torch.nn.Sequential(*layers)

        for i, dim in enumerate(self.dims):
            # +1 for bias
            # prev_layer for previous layer
            # input dimension per neuron
            if i > 0:
                prev_layer = inside_dims[-1] * dim
            self.layers.append(make_mini(prev_layer + 1 + dim))

        self.layers = torch.nn.ModuleList(self.layers)
        
        # Final network to combine them all together
        self.rho = nn.Linear(inside_dims[-1] * len(dims), n_classes)

    def forward(self, params, getrep=False):
        reps = []
        prev_layer_reps = None
        is_batched = len(params[0].shape) > 2

        for param, layer in zip(params, self.layers):

            # Case where data is batched per layer
            if is_batched:
                if prev_layer_reps is None:
                    param_eff = param
                else:
                    prev_layer_reps = prev_layer_reps.repeat(1, param.shape[1], 1)
                    param_eff = torch.cat((param, prev_layer_reps), -1)

                prev_shape = param_eff.shape
                processed = layer(param_eff.view(-1, param_eff.shape[-1]))
                processed = processed.view(prev_shape[0], prev_shape[1], -1)

            else:
                if prev_layer_reps is None:
                    param_eff = param
                else:
                    prev_layer_reps = prev_layer_reps.repeat(param.shape[0], 1)
                    # Include previous layer representation
                    param_eff = torch.cat((param, prev_layer_reps), -1)
                processed = layer(param_eff)

            # Store this layer's representation
            reps.append(torch.sum(processed, -2))
            if is_batched:
                prev_layer_reps = processed.view(processed.shape[0], -1)
            else:
                prev_layer_reps = processed.view(-1)
            prev_layer_reps = torch.unsqueeze(prev_layer_reps, -2)

        if is_batched:
            reps_c = torch.cat(reps, 1)
        else:
            reps_c = torch.unsqueeze(torch.cat(reps), 0)

        if getrep:
            return reps_c
        logits = self.rho(reps_c)
        return logits

##### CENSUS ######

def save_model_census(clf, path):
    dump(clf, path)

def load_model_census(path):
    return load(path)


def get_models_path(path, property, split, value=None):
    if value is None:
        return os.path.join(path, property, split)
    return os.path.join(path,  property, split, value)

def get_model(max_iter=500,hidden_layer_sizes=(32, 16, 8),):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,max_iter=max_iter)
    return clf


##### BONEAGE #####
def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_if_exists(result_path, model_id, split, ratio):

    model_check_path = os.path.join(result_path, split, str(ratio))
    # Return false if no directory exists
    if not os.path.exists(model_check_path):
        return False
    # If it does, check for models inside
    for model_name in os.listdir(model_check_path):
        if model_name.startswith("%d_" % model_id):
            return True
    return False

# Save model in specified directory
def save_model_boneage(result_path, model, split, ratio, prop_and_name):
    subfolder_prefix = os.path.join(split, str(ratio))
    # if full_model:
    #     subfolder_prefix = os.path.join(subfolder_prefix, "full")

    # Make sure directory exists
    ensure_dir_exists(os.path.join(result_path, subfolder_prefix))

    torch.save(model.state_dict(), os.path.join(result_path, subfolder_prefix, prop_and_name))


# Load model from given directory
def load_model_boneage(path: str, fake_relu: bool = False,latent_focus: int = None,cpu: bool = False):

    model = BoneModel(1024, fake_relu=fake_relu, latent_focus=latent_focus)
    if cpu:
       model_dict = torch.load(path, map_location=torch.device("cpu"))
    else:
       model_dict = torch.load(path)

    if "actual_model" in model_dict:
        # Information about training data also stored; return
        model.load_state_dict(model_dict["actual_model"])
        train_ids = model_dict["train_ids"]
        test_ids = model_dict["test_ids"]
        # return model, (train_ids, test_ids)
    else:
        model.load_state_dict(model_dict)

    model.eval()
    return model

# fake relu function
class fakerelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # return input.clamp(min=0)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# identity function module wrapper
class BasicWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(BasicWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor):
        return fakerelu.apply(input)

class FakeReluWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(FakeReluWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor):
        return fakerelu.apply(input)


class BoneModel(nn.Module):
    def __init__(self,n_inp: int,fake_relu: bool = False,latent_focus: int = None):
        if latent_focus is not None:
            if latent_focus not in [0, 1]:
                raise ValueError("Invalid interal layer requested")

        if fake_relu:
            act_fn = BasicWrapper
        else:
            act_fn = nn.ReLU

        self.latent_focus = latent_focus
        super(BoneModel, self).__init__()
        layers = [nn.Linear(n_inp, 128),FakeReluWrapper(inplace=True),nn.Linear(128, 64),FakeReluWrapper(inplace=True),nn.Linear(64, 1)]

        mapping = {0: 1, 1: 3}
        if self.latent_focus is not None:
            layers[mapping[self.latent_focus]] = act_fn(inplace=True)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, latent: int = None) -> torch.Tensor:
        if latent is None:
            return self.layers(x)

        if latent not in [0, 1]:
            raise ValueError("Invalid interal layer requested")

        latent = (latent * 2) + 1

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == latent:
                return x




######################################## Arxiv Dataset ################################

from dgl.nn.pytorch import GraphConv
import torch as ch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    def __init__(self, ds, n_hidden, n_layers,dropout):
        super(GCN, self).__init__()
        # Get actual graph
        self.g = ds.g

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(ds.num_features, n_hidden, activation=F.relu))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=F.relu))

        # output layer
        self.layers.append(GraphConv(n_hidden, ds.num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, latent=None):
        if latent is not None:
            if latent < 0 or latent > len(self.layers):
                raise ValueError("Invald interal layer requested")

        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i == latent:
                return h
        return h


def get_model_arxiv(ds, args):
    model = GCN(ds, args.hidden_channels,args.num_layers, args.dropout)
    model = model.to(device)
    return model


def save_model_arxiv(model, savepath):
    torch.save(model.state_dict(), savepath)


def train_arxiv(model, ds, train_idx, optimizer, loss_fn):
    model.train()

    X = ds.get_features()
    Y = ds.get_labels()

    optimizer.zero_grad()
    out = model(X)[train_idx]
    loss = loss_fn(out, Y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_arxiv(model, ds, train_idx, test_idx, evaluator, return_preds = False):
    model.eval()

    X = ds.get_features()
    Y = ds.get_labels()

    out = model(X)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({'y_true': Y[train_idx],'y_pred': y_pred[train_idx]})['acc']
    test_acc = evaluator.eval({'y_true': Y[test_idx],'y_pred': y_pred[test_idx]})['acc']

    if return_preds:
        return train_acc, test_acc, y_pred
    return train_acc, test_acc


def train_model_arxiv(ds, model, evaluator, args):
    run_accs = {"train": [],"test": []}

    train_idx, test_idx = ds.get_idx_split()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    iterator = tqdm(range(1, 1 + args.epochs))

    for epoch in iterator:
        loss = train_arxiv(model, ds, train_idx, optimizer, loss_fn)
        train_acc, test_acc = test_arxiv(model, ds, train_idx, test_idx, evaluator)

        iterator.set_description(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_acc:.2f}%, Test: {100 * test_acc:.2f}%')

        run_accs["train"].append(train_acc)
        run_accs["test"].append(test_acc)

    return run_accs


def get_weight_layers_arxiv(m, normalize=False, transpose=True,
                      first_n=np.inf, start_n=0,
                      custom_layers=None,
                      conv=False, include_all=False):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0

    # Sort and store desired layers, if specified
    custom_layers = sorted(custom_layers) if custom_layers is not None else None

    for name, param in m.named_parameters():
        if "weight" in name:
            param_data = param.data.detach().cpu()
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

        # Assume each layer has weight & bias
        i += 1

        if custom_layers is None:
            # If requested, start looking from start_n layer
            if (i - 1) // 2 < start_n:
                dims, dim_kernels, weights, biases = [], [], [], []
                continue

            # If requested, look at only first_n layers
            if i // 2 > first_n - 1:
                break
        else:
            # If this layer was not asked for, omit corresponding weights & biases
            if i // 2 != custom_layers[j // 2]:
                dims = dims[:-1]
                dim_kernels = dim_kernels[:-1]
                weights = weights[:-1]
                biases = biases[:-1]
            else:
                # Specified layer was found, increase count
                j += 1

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


def verify_arxiv_gnn_performance(args, ds, clfs, model_dir):
    all_accs = []
    print("Nodes: %d, Average degree: %.2f" %(ds.num_nodes, ds.g.number_of_edges() / ds.num_nodes))
    print("Train: %d, Test: %d" % (len(ds.train_idx), len(ds.test_idx)))

    train_idx, test_idx = ds.get_idx_split()
    evaluator = Evaluator(name='ogbn-arxiv')

    print("Going into testing loop")
    for clf in tqdm(clfs):
        clf = clf.to('cpu') # Need for now since cuda is not working with Arxiv yet
        clf.eval()
        _, avg_acc = test_arxiv(clf, ds, train_idx, test_idx, evaluator)
        all_accs.append(avg_acc)
    
    mean_acc = statistics.mean(all_accs)
    std_dev_acc = statistics.stdev(all_accs)

    print(f'Mean accuracy of loaded {args.dataset} model is: {mean_acc}')
    print(f'Standard deviation of accuracy for loaded {args.dataset} model is: {std_dev_acc}')



def get_model_features_arxiv(model_dir, ds, args, max_read=None, verify_model = False, fetch_models=False):
    vecs = []
    iterator = os.listdir(model_dir)
    clfs = []
    if max_read is not None:
        np.random.shuffle(iterator)
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        # Define model
        model = get_model_arxiv(ds, args)

        # Load weights into model
        # if args.gpu:
        #     model.load_state_dict(torch.load(os.path.join(model_dir, mpath)))
        # else:
        model.load_state_dict(torch.load(os.path.join(model_dir, mpath), map_location=device))
        model.eval()
        clfs.append(model)

        # Extract model weights
        dims, fvec = get_weight_layers_arxiv(
            model, transpose=False, first_n=args.first_n,
            start_n=args.start_n)

        vecs.append(fvec)
    
    if verify_model:
        verify_arxiv_gnn_performance(args, ds, clfs, model_dir)

    return dims, vecs, clfs