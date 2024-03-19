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
import torch
from pathlib import Path
import random
from . import meta_utils
from . import os_layer
from . import models

# from foolbox import PyTorchModel, accuracy, samples
# from foolbox.attacks import LinfPGD, LinfFastGradientAttack
# from foolbox.criteria import TargetedMisclassification
import gc
from autoattack import AutoAttack


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_attack_weights_into_basemodel(args, model, x_advs, result_path):
    if args.dataset == 'CENSUS':
        weights = [torch.from_numpy(x) for x in model.coefs_]
        biases = [torch.from_numpy(x) for x in model.intercepts_]
        
        # Performance prior to weight replacement
        print()
        print('Base model performance prior to weight replacement')
        model_path = meta_utils.get_models_path(result_path,"prover/test", args.filter + "", args.claim_ratio)
        meta_utils.verify_basemodel_performance(args, [model], model_path)

        # Performance after
        if len(x_advs[0].shape) == 3:
            transposed_first_model_params = x_advs[0][0].squeeze(0).T.cpu().numpy()
        else:
            # shape len 2
            transposed_first_model_params = x_advs[0].T.cpu().numpy()
        first_model_weights = transposed_first_model_params[:weights[0].shape[0], :]
        first_model_biases = transposed_first_model_params[weights[0].shape[0]:, :]
       
        if (model.coefs_[0] == first_model_weights).all():
            print('Perturbed and clean weights are the same. Something may be incorrect')
        else:
            print('Initial weights are different, as expected')
        
        if (model.intercepts_[0] == first_model_biases).all():
            print('Perturbed and clean weights are the same. Something may be incorrect')
        else:
            print('Initial weights are different, as expected')
        
        model.coefs_[0] = first_model_weights
        model.intercepts_[0] = first_model_biases

        print()
        print('Base model performance following weight replacement')
        meta_utils.verify_basemodel_performance(args, [model], model_path)
        print()
        return model, first_model_weights, first_model_biases

    else:
        model_path = os.path.join(result_path, "prover/test/%s" % args.claim_ratio)
        if args.dataset == 'BONEAGE':
            transpose = True
            prune_mask = []
            _, fvec = meta_utils.get_weight_layers(model, first_n=1, start_n=0, prune_mask=prune_mask)

        elif args.dataset == 'ARXIV':
            transpose = False
            model_path = os.path.join(result_path, "prover/test",  args.claim_degree)
            _, fvec = models.get_weight_layers_arxiv(
                model, transpose=transpose, first_n=1,
                start_n=0)

        # Performance prior to weight replacement
        print('Base model performance prior to weight replacement')
        ds_made = meta_utils.verify_basemodel_performance(args, [model], model_path)

        # if not transpose:
        print(f'x_advs[0] shape: {x_advs[0].shape}')
        if len(x_advs[0].shape) == 3:
            params_to_take = x_advs[0][0].squeeze(0).T # [first_batch][first_idx_in_batch].squeeze_out_dim_0
        else:
            params_to_take = x_advs[0].T
        print(f'params_to_take shape: {params_to_take.shape}')

        weights = params_to_take[:params_to_take.shape[0]-1, :]
        biases = params_to_take[params_to_take.shape[0]-1:, :].squeeze(0)

        print(f'weights shape: {weights.shape}')
        print(f'biases shape: {biases.shape}')

        state_dict = model.state_dict()
        print('Checking if the current params differ')
        if args.dataset == 'ARXIV':
            if (state_dict['layers.0.weight'] == weights.cpu()).all():
                print('Perturbed and clean weights are the same. Something may be incorrect')
            else:
                print('Initial weights are different, as expected')
        elif args.dataset =='BONEAGE':
            if (state_dict['layers.0.weight'] == weights.T.to(device)).all():
                print('Perturbed and clean weights are the same. Something may be incorrect')
            else:
                print('Initial weights are different, as expected')
        
        if (state_dict['layers.0.bias'].cuda() == biases.to(device)).all():
            print('Perturbed and clean weights are the same. Something may be incorrect')
        else:
            print('Initial weights are different, as expected')

        if args.dataset == 'BONEAGE':
            state_dict['layers.0.weight'] = weights.T
            state_dict['layers.0.bias'] = biases
        else:
            state_dict['layers.0.weight'] = weights
            state_dict['layers.0.bias'] = biases

        model.load_state_dict(state_dict)

        # Freezing first layer
        for name, param in model.named_parameters():
            if name == 'layers.0.weight' or name == 'layers.0.bias':
                param.requires_grad = False
                print(f'{name} is frozen')

        # Model performance after
        print('Base model performance following weight replacement')
        if args.dataset == 'BONEAGE' or (args.dataset == 'ARXIV' and ds_made is not None):
            meta_utils.verify_basemodel_performance(args, [model], model_path, ds=ds_made)
        
        return model, weights, biases


def choose_basemodels(args, x_advs, label_vals, clfs):
    print(f'len(x_advs): {len(x_advs)}, len(label_vals): {len(label_vals)}, len(clfs): {len(clfs)}')
    try:
        assert(len(x_advs) == len(label_vals))
    except:
        raise Exception(f'len(x_advs): {len(x_advs)}, len(label_vals): {len(label_vals)}')
    try:
        assert(len(clfs) == len(label_vals))
    except:
        raise Exception(f'len(clfs): {len(clfs)}, len(label_vals): {len(label_vals)}')

    distinct_vals = list(set(label_vals))
    dict_per_val = {round(dv.item(),2):[] for dv in distinct_vals} # ratio -> [(x_adv, clf)]
    
    for xa, lv, clf in zip(x_advs, label_vals, clfs):
        dict_per_val[round(lv.item(),2)].append((xa, clf))
    
    min_number_clfs_to_take = min(args.num_finetuned_basemodels, len(min(dict_per_val.items(), key=lambda x: len(x[1]))[1]))

    for k, v in dict_per_val.items():
        dict_per_val[k] = random.sample(v, min_number_clfs_to_take)
    
    # print(dict_per_val)
    return dict_per_val


def generate_and_save_perturbed_data(args, rep_gen, attestation_thresh, X_train_adv, Y_train, X_test, Y_test, adv_samples_path, log):
    x_advs_train = do_attack(args, rep_gen.cuda(), X_train_adv, y = Y_train, eps=8/255, attestation_thresh = attestation_thresh)
    x_advs_train = torch.cat(x_advs_train, dim=0).squeeze(1)
    # X_train_rob = torch.cat((X_train_adv, x_advs_train))
    # Y_train_rob = torch.cat((Y_train, Y_train[:len(X_train) // 2]))
    property_val = args.claim_ratio
    if args.dataset == 'ARXIV':
        property_val = args.claim_degree

    if args.dataset == 'CENSUS':
        train_advs_path = Path(os.path.join(adv_samples_path, args.dataset, "verifier", args.filter))
    else:
        train_advs_path = Path(os.path.join(adv_samples_path, args.dataset, "verifier"))
    os_layer.create_dir_if_doesnt_exist(train_advs_path, log)

    torch.save({'data':x_advs_train[Y_train==0], 'labels':Y_train[Y_train==0]}, train_advs_path / f'{property_val}.pt')
    print(f'Saved perturbed training data at: {train_advs_path} / {property_val}.pt')
    x_advs_train = None
    gc.collect()

    x_advs_test = do_attack(args, rep_gen.cuda(), X_test, y = Y_test, eps=8/255, attestation_thresh = attestation_thresh)
    print(f'type of x_advs_test: {type(x_advs_test)}')
    print(f'len of x_advs_test: {len(x_advs_test)}')
    x_advs_test = torch.cat(x_advs_test, dim=0).squeeze(1)
    # X_test_rob = torch.cat((X_test, x_advs_test))
    # Y_test_rob = torch.cat((Y_test, Y_test[:len(X_test) // 2]))
    if args.dataset == 'CENSUS':
        test_advs_path = Path(os.path.join(adv_samples_path, args.dataset, "prover", args.filter))
    else:
        test_advs_path = Path(os.path.join(adv_samples_path, args.dataset, "prover"))
    os_layer.create_dir_if_doesnt_exist(test_advs_path, log)
    torch.save({'data':x_advs_test[Y_test==0], 'labels':Y_test[Y_test==0]}, test_advs_path / f'{property_val}.pt')
    print(f'Saved perturbed test data at {test_advs_path} / {property_val}.pt')
    x_advs_test = None
    gc.collect()


def check_basemodel_performance_on_perturbed(args, model_dict, result_path, adv_path):
    if not isinstance(model_dict, dict):
        raise Exception("Please use a dictionary mapping ratio -> (x_adv, clfs)")
    
    attacked_basemodel_dict = {k:[] for k in model_dict.keys()} # ratio -> [(attacked_model, weights, biases)]

    for ratio, adv_clf_tuples in model_dict.items():
        print(f'Conducting finetuned performance check for ratio {ratio}')
        for i, adv_clf_tuple in enumerate(adv_clf_tuples):
            attacked_basemodel, weights, biases = load_attack_weights_into_basemodel(args, adv_clf_tuple[1], [adv_clf_tuple[0]], result_path)
            if args.dataset == 'BONEAGE':
                weights = weights.T

            # Let's see if we can improve the attacked_basemodel's performance on it's data
            model_path = os.path.join(result_path, "prover/test/%s" % args.claim_ratio)
            attacked_basemodel = meta_utils.verify_basemodel_performance(args, [attacked_basemodel], model_path, finetune = True, weights=weights, biases=biases)
            
            if args.dataset == 'BONEAGE':
                models.save_model_boneage(adv_path, attacked_basemodel[0], os.path.join("prover","test", f'claim_{args.claim_ratio}'), ratio,"%d.pth" %(i))

            attacked_basemodel_dict[ratio].append((attacked_basemodel[0], weights, biases))
            print()

    return attacked_basemodel_dict


def prepare_defense_data(args, adv_samples_path, X_train, Y_train, X_test, Y_test):
    if args.dataset == 'CENSUS':
        train_advs_path = Path(os.path.join(adv_samples_path, args.dataset, "verifier", args.filter))
        test_advs_path = Path(os.path.join(adv_samples_path, args.dataset, "prover", args.filter))
    else:
        train_advs_path = Path(os.path.join(adv_samples_path, args.dataset, "verifier"))
        test_advs_path = Path(os.path.join(adv_samples_path, args.dataset, "prover"))

    property_val = args.claim_ratio
    if args.dataset == 'ARXIV':
        property_val = args.claim_degree

    adv_train_data = torch.load(train_advs_path / f'{property_val}.pt')
    adv_test_data = torch.load(test_advs_path / f'{property_val}.pt')

    x_advs_train = adv_train_data['data']
    x_advs_test = adv_test_data['data']
    y_advs_train = adv_train_data['labels']
    y_advs_test = adv_test_data['labels']
    print(f'y_advs_train: {y_advs_train}')
    print(f'y_advs_test: {y_advs_test}')


    # Just take 10% of the saved data (10% of 50%, so 5% of total)
    # x_advs_train = x_advs_train[:len(x_advs_train) // 10]
    # x_advs_test = x_advs_test[:len(x_advs_test) // 10]
    # y_advs_train = y_advs_train[:len(y_advs_train) // 10]
    # y_advs_test = y_advs_test[:len(y_advs_test) // 10]

    gc.collect()

    print(f'X_train[0] shape before: {X_train[0].shape}')
    print(f'Y_train shape before: {Y_train.shape}')
    print(f'X_test[0] shape before: {X_test[0].shape}')
    print(f'Y_test shape before: {Y_test.shape}')
    print()

    if args.dataset == 'BONEAGE':
        X_train = torch.cat((X_train, x_advs_train.cpu()))
        Y_train = torch.cat((Y_train, y_advs_train))

        X_test = torch.cat((X_test, x_advs_test.cpu()))
        Y_test = torch.cat((Y_test, y_advs_test))
    else:
        X_train = [torch.cat((X_train[0], x_advs_train))]
        Y_train = torch.cat((Y_train, y_advs_train))

        X_test = [torch.cat((X_test[0], x_advs_test))]
        Y_test = torch.cat((Y_test, y_advs_test))

    print(f'X_train[0] shape after: {X_train[0].shape}')
    print(f'Y_train shape after: {Y_train.shape}')
    print(f'X_test[0] shape after: {X_test[0].shape}')
    print(f'Y_test shape after: {Y_test.shape}')

    return X_train, Y_train, X_test, Y_test


def do_attack(args, model, x, y, actual_ratio_labels = None, attestation_thresh = None, batch_size=8, device='cuda', eps=2/255, use_logits = True):
    model.eval()
    model.aucroc_thresh = attestation_thresh
    model.use_logits = use_logits
    print(f'Using eps = {eps}')
    print(f'shape of x is :{x.shape}') 
    print(f'shape of y is: {y.shape}')
    # apgd-t, fab-t, square (these are the others besides apgd-ce) for attacks_to_run. Only apgd-ce and square apply since the other two have 9 target classes
    adversary = AutoAttack(model, norm='Linf', eps=eps, device=device, verbose=True, version='custom', attacks_to_run=['apgd-ce', 'square'])
    x = x.unsqueeze(1) # simulate single channel

    x_advs = []
    for i in range(1, len(y)):
        x_to_use = x[(i-1)*batch_size:i*batch_size].to(device)
        y_to_use = y[(i-1)*batch_size:i*batch_size].type(torch.LongTensor).to(device)
        print(f'len of x_to_use is: {len(x_to_use)}')
        if len(x_to_use) == 0:
            print('Finished')
            break
        x_adv = adversary.run_standard_evaluation(x_to_use, y_to_use, bs=x_to_use.shape[0])
        x_advs.append(x_adv.cpu())
        print('Done a run')

    model.use_logits = False
    return x_advs


