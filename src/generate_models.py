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

import argparse
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import os
import torch
from ogb.nodeproppred import Evaluator
from timeit import Timer

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from . import data
from . import models
from . import os_layer
from . import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace, log: logging.Logger) -> None:

    raw_path: Path = Path(args.raw_path)
    raw_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(raw_path, log)
    result_path: Path = Path(args.result_path)
    result_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(result_path, log)
    suff = str(args.ratio)
    if args.dataset == 'CENSUS':
        suff = args.filter +"/"+ suff
    elif args.dataset == 'ARXIV':
        suff = str(args.degree)
    if args.test:
        suff = "test" + "/"+suff
    sampled_save_path: Path = Path(args.sampled_dataset_save_path) / args.dataset / args.split / suff
    sampled_save_path_status, sampled_save_path_already_exists = os_layer.create_dir_if_doesnt_exist(sampled_save_path, log)
    if (raw_path_status or result_path_status or sampled_save_path_status) is None:
        msg: str = f"Something went wrong when creating {raw_path} or {result_path} or {sampled_save_path}. Aborting..."
        log.error(msg)
        raise EnvironmentError(msg)
    log.info("Dataset {}".format(args.dataset))


    if args.dataset == "CENSUS":
        raw_path = raw_path / "census_splits"
        result_path = result_path / "CENSUS"
        dataset = data.CensusWrapper(path=raw_path,filter_prop=args.filter, ratio=float(args.ratio), split=args.split)
        dataset_prover = None
        if args.overlap > float(0):
            dataset_prover = data.CensusWrapper(path=raw_path,filter_prop=args.filter, ratio=float(args.ratio), split="prover")
        

        for i in tqdm(range(1, args.num + 1)):
            print("Training classifier %d" % i)

            # if sampled_save_path_already_exists:
            #     (x_tr, y_tr), (x_te, y_te), _, _ = data.load_saved_traintest_data(args, sampled_save_path)
            # else:
            # Putting it back inside
            if dataset_prover is not None:
                _, _, _, train_df_prover, test_df_prover = dataset_prover.load_data(send_original_dfs=True)
                (x_tr, y_tr), (x_te, y_te), cols, train_df, test_df = data.generate_overlap_datasets(args, dataset, dataset_prover)
            else:
                (x_tr, y_tr), (x_te, y_te), cols, train_df, test_df = dataset.load_data(send_original_dfs=True)

            data.save_sampled_dataset(args, sampled_save_path, (x_tr, y_tr), (x_te, y_te), training_df_census = train_df, testing_df_census = test_df, suffix = i)
            print(f'x_tr shape: {x_tr.shape}')
            print(f'x_te shape: {x_te.shape}')

            clf = models.get_model()
            clf.fit(x_tr, y_tr.ravel())
            train_acc = 100 * clf.score(x_tr, y_tr.ravel())
            test_acc = 100 * clf.score(x_te, y_te.ravel())
            print("Classifier %d : Train acc %.2f , Test acc %.2f\n" % (i, train_acc, test_acc))

            if args.test:
                save_path = models.get_models_path(result_path, os.path.join(args.split,"test"), args.filter, str(args.ratio))
            else:
                save_path = models.get_models_path(result_path, args.split, args.filter, str(args.ratio))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            print("Saving model to %s" % os.path.join(save_path, str(i) + "_%.2f" % test_acc))
            models.save_model_census(clf, os.path.join(save_path,str(i) + "_%.2f" % test_acc))

    elif args.dataset == "BONEAGE":
        result_path = result_path / "BONEAGE"
        raw_path = raw_path / "boneage_splits"

        def filter1(x): return x["gender"] == 1

        df_train, df_val = data.get_df(raw_path, args.split)
        df_train_prover, df_val_prover = None, None
        features = data.get_features(raw_path, args.split)
        features_prover = None
        if args.overlap > float(0):
            df_train_prover, df_val_prover = data.get_df(raw_path, "prover")
            features_prover = data.get_features(raw_path, "prover")
            features['train'] = {**features['train'], **features_prover['train']}
            features['val'] = {**features['val'], **features_prover['val']}

        # print(f'features are: {features}')

        n_train, n_test = 700, 200
        if args.split == "prover":
            n_train, n_test = 1400, 400

        
        for i in range(args.num):
            if models.check_if_exists(result_path, i+1, args.split, args.ratio):
                continue
            # if sampled_save_path_already_exists:
            #     df_train_processed, df_val_processed = data.load_saved_traintest_data(args, sampled_save_path)
            # else:
            if df_train_prover is not None:
                df_train_processed, df_val_processed = data.generate_overlap_datasets(args, (df_train, df_val), (df_train_prover, df_val_prover))
            else:
                df_train_processed = data.heuristic(df_train, filter1, args.ratio,n_train, class_imbalance=1.0, n_tries=300)
                df_val_processed = data.heuristic(df_val, filter1, args.ratio,n_test, class_imbalance=1.0, n_tries=300)

            data.save_sampled_dataset(args, sampled_save_path, df_train_processed, df_val_processed, suffix=i)

            model = models.BoneModel(1024).to(device)
            ds = data.BoneWrapper(df_train_processed,df_val_processed,features=features,augment=False)
            train_loader, val_loader = ds.get_loaders(256*32, shuffle=True)
            _, vacc = utils.train(model, (train_loader, val_loader),lr=1e-3,epoch_num=100,weight_decay=1e-4)

            if args.test:
                model_dir_path = os.path.join(result_path, args.split, "test", str(args.ratio))
            else:
                model_dir_path = os.path.join(result_path, args.split, str(args.ratio))

            if not os.path.exists(model_dir_path):
                os.makedirs(model_dir_path)

            if args.test:
                models.save_model_boneage(result_path, model, os.path.join(args.split,"test"), args.ratio,"%d_%.2f.pth" %(i+1, vacc*100))
            else:
                models.save_model_boneage(result_path, model, args.split, args.ratio,"%d_%.2f.pth" %(i+1, vacc*100))
            print("[Status] %d/%d" % (i+1, args.num))

    elif args.dataset == "ARXIV":
        result_path = result_path / "ARXIV"
       
        assert args.prune >= 0 and args.prune <= 0.1 # Prune ratio should be valid and not too large
        ds = data.ArxivNodeDataset(args.split)
        # Need to save this one

        n_edges = ds.g.number_of_edges()
        n_nodes = ds.g.number_of_nodes()
        print("""----Data statistics------'\n #Edges %d; #Nodes %d; #Classes %d""" %(n_edges, n_nodes, ds.num_classes,))

        print("-> Before modification")
        print("Nodes: %d, Average degree: %.2f" %(ds.num_nodes, ds.g.number_of_edges() / ds.num_nodes))
        print("Train: %d, Test: %d" % (len(ds.train_idx), len(ds.test_idx)))

        ds.change_mean_degree(args.degree, args.prune)

        pick_tr, pick_te = 30000, 10000
        if args.split == "prover":
            pick_tr, pick_te = 62000, 35000

        ds.label_ratio_preserving_pick(pick_tr, pick_te)

        print("-> After modification")
        print("Nodes: %d, Average degree: %.2f" %(ds.num_nodes, ds.g.number_of_edges() / ds.num_nodes))
        print("Train: %d, Test: %d" % (len(ds.train_idx), len(ds.test_idx)))

        for index in range(args.num):
            model = models.get_model_arxiv(ds, args)
            evaluator = Evaluator(name='ogbn-arxiv')

            run_accs = models.train_model_arxiv(ds, model, evaluator, args)
            acc_tr, acc_te = run_accs["train"][-1], run_accs["test"][-1]
            print("Train accuracy: {:.2f}; Test accuracy: {:.2f}".format(acc_tr*100,acc_te*100))

            if args.test:
                savepath = result_path / args.split / "test" / str(int(args.degree))
            else:
                savepath = result_path / args.split / str(int(args.degree)) 
            savepath_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(savepath, log)
            models.save_model_arxiv(model, savepath / "{}_{:.2f}.pth".format(index,acc_tr*100))

    else:
        msg_dataset_error: str = "No such dataset"
        log.error(msg_dataset_error)
        raise EnvironmentError(msg_dataset_error)


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--raw_path', type=str, default="dataset/", help='Root directory of the dataset')
    parser.add_argument('--result_path', type=str, default="results_overlap_25/", help='Results directory for the models')
    parser.add_argument('--dataset', type=str, default="CENSUS", help='Options: CENSUS, BONEAGE, ARXIV')
    parser.add_argument('--sampled_dataset_save_path', type=str, default="sampled_datasets/", help='Directory location of where to save the training and testing data')
    parser.add_argument('--filter', type=str,required=False,help='while filter to use')
    parser.add_argument('--ratio', type=float, default=0.5,help='what ratio of the new sampled dataset')
    parser.add_argument('--num', type=int, default=100,help='how many classifiers to train?')
    parser.add_argument('--split', choices=["verifier", "prover"],required=True,help='which split of data to use')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--overlap', type=float, default=0., help='How much overlap to use between the prover and verifier dataset')

    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--degree', type=float, default=13)
    parser.add_argument('--prune', type=float, default=0)

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="generate_models.log", filemode="w")
    log: logging.Logger = logging.getLogger("GenerateModels")
    args = handle_args()

    if args.overlap > float(0):
        assert(args.split=='verifier')

    # execute code segment and find the execution time
    t = Timer(lambda: main(args, log))
    exec_time = t.timeit(number=1)
    
    # printing the execution time in secs.
    # nearest to 2-decimal places
    print(f"Total base classifier training time: {exec_time:.02f} secs.")
