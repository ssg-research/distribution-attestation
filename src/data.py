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
from sklearn.model_selection import StratifiedShuffleSplit
import requests
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch



def filter(df, condition, ratio, verbose=True):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))
    # If current ratio less than desired ratio, subsample from non-ratio
    if verbose:
        print("Changing ratio from %.2f to %.2f" % (current_ratio, ratio))
    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[:int(((1-ratio) * len(qualify))/ratio)]
            return pd.concat([df.iloc[qualify], df.iloc[nqi]])
        return df.iloc[qualify]
    else:
        np.random.shuffle(qualify)
        if ratio > 0:
            qi = qualify[:int((ratio * len(notqualify))/(1 - ratio))]
            return pd.concat([df.iloc[qi], df.iloc[notqualify]])
        return df.iloc[notqualify]


def heuristic(df, condition, ratio, cwise_sample, class_imbalance=2.0,n_tries=1000,class_col="label",verbose=True):
    vals, pckds = [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        pckd_df = filter(df, condition, ratio, verbose=False)

        # Class-balanced sampling
        zero_ids = np.nonzero(pckd_df[class_col].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df[class_col].to_numpy() == 1)[0]

        # Sub-sample data, if requested
        if cwise_sample is not None:
            if class_imbalance >= 1:
                zero_ids = np.random.permutation(zero_ids)[:int(class_imbalance * cwise_sample)]
                one_ids = np.random.permutation(one_ids)[:cwise_sample]
            else:
                zero_ids = np.random.permutation(zero_ids)[:cwise_sample]
                one_ids = np.random.permutation(one_ids)[:int(1 / class_imbalance * cwise_sample)]

        # Combine them together
        pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
        pckd_df = pckd_df.iloc[pckd]

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description("%.4f" % (ratio + np.min([np.abs(zz-ratio) for zz in vals])))

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    return picked_df.reset_index(drop=True)


# This is unused
def load_saved_traintest_data(args, save_path, suffix=''):
    if args.dataset == 'BONEAGE':
        train_df = pd.read_csv(str(save_path),f"train_{suffix}.csv")
        test_df = pd.read_csv(str(save_path),f"test_{suffix}.csv")
        return train_df, test_df
    elif args.dataset == 'CENSUS':
        with open(os.path.join(save_path, f"train_{suffix}.npy"), 'rb') as f:
            train_x = np.load(f)
            train_y = np.load(f)
        with open(os.path.join(save_path, f"test_{suffix}.npy"), 'rb') as f:
            test_x = np.load(f)
            test_y = np.load(f)
        training_census_df = pd.read_csv(os.path.join(str(save_path),f'training_census_df_{suffix}.csv'))
        testing_census_df = pd.read_csv(os.path.join(str(save_path),f'testing_census_df_{suffix}.csv'))
        return (train_x, train_y), (test_x, test_y), training_census_df, testing_census_df


def save_sampled_dataset(args, save_path, training_set, testing_set, training_df_census = None, testing_df_census = None, suffix=''):
    # Save the dfs
    if args.dataset == 'BONEAGE':
        training_set.to_csv(os.path.join(str(save_path),f"train_{suffix}.csv"))
        testing_set.to_csv(os.path.join(str(save_path),f"test_{suffix}.csv"))
    elif args.dataset == 'CENSUS':
        with open(os.path.join(save_path, f"train_{suffix}.npy"), 'wb') as f:
            np.save(f, training_set[0])
            np.save(f, training_set[1])
        with open(os.path.join(save_path, f"test_{suffix}.npy"), 'wb') as f:
            np.save(f, testing_set[0])
            np.save(f, testing_set[1])
        training_df_census.to_csv(os.path.join(str(save_path), f'training_census_df_{suffix}.csv'))
        testing_df_census.to_csv(os.path.join(str(save_path), f'testing_census_df_{suffix}.csv'))

    print(f'Datasets saved')


def return_overlapped_set(args, verif_set, prover_set):
    verif_set_sampled = verif_set.sample(frac=(1-args.overlap))
    prover_set_sampled = prover_set.sample(frac=args.overlap)

    final_df = verif_set_sampled.append(prover_set_sampled, ignore_index = True)
    return final_df


def returned_partitioned_sets(df, feature):
    df_portion1 = df.loc[df[feature]==1]
    df_portion2 = df.loc[df[feature]==0]
    return df_portion1, df_portion2


def generate_overlap_datasets(args, dataset, dataset_prover):
    assert(args.overlap > float(0))
    assert(args.dataset != 'ARXIV') # Not supported for Arxiv at the moment
    
    if args.dataset =='CENSUS':
        _, _, _, train_df_prover, test_df_prover = dataset_prover.load_data(send_original_dfs=True)
        _, _, _, train_df, test_df = dataset.load_data(send_original_dfs=True)

        feature = 'sex:Male'
        if args.filter == 'race':
            feature = 'race:White'        

    
    if args.dataset =='BONEAGE':
        (df_train, df_test), (df_train_prover, df_val_prover) = dataset, dataset_prover

        n_train, n_test = 700, 200
        n_train_prover, n_test_prover = 1400, 400

        def filter1(x): return x["gender"] == 1

        train_df = heuristic(df_train, filter1, args.ratio, n_train, class_imbalance=1.0, n_tries=300)
        test_df = heuristic(df_test, filter1, args.ratio, n_test, class_imbalance=1.0, n_tries=300)

        train_df_prover = heuristic(df_train_prover, filter1, args.ratio, n_train_prover, class_imbalance=1.0, n_tries=300)
        test_df_prover = heuristic(df_val_prover, filter1, args.ratio, n_test_prover, class_imbalance=1.0, n_tries=300)

        feature='gender'
    

    # No longer doing overlap for the testing set
    train_df_portion1, train_df_portion2 = returned_partitioned_sets(train_df, feature)
    # test_df_portion1, test_df_portion2 = returned_partitioned_sets(test_df, feature)

    train_df_prover_portion1, train_df_prover_portion2 = returned_partitioned_sets(train_df_prover, feature)
    # test_df_prover_portion1, test_df_prover_portion2 = returned_partitioned_sets(test_df_prover, feature)
    

    # Sample args.overlap of the prover dataset, and 1-overlap of the verifier dataset, and combine them
    # Do this for each portion
    train_df_final_portion1 = return_overlapped_set(args, train_df_portion1, train_df_prover_portion1)
    train_df_final_portion2 = return_overlapped_set(args, train_df_portion2, train_df_prover_portion2)
    TRAIN_DF = train_df_final_portion1.append(train_df_final_portion2, ignore_index=True)

    # test_df_final_portion1 = return_overlapped_set(args, test_df_portion1, test_df_prover_portion1)
    # test_df_final_portion2 = return_overlapped_set(args, test_df_portion2, test_df_prover_portion2)
    # TEST_DF = test_df_final_portion1.append(test_df_final_portion2, ignore_index=True)

    if args.dataset == 'CENSUS':
        (x_tr, y_tr, cols), (x_te, y_te, cols) = dataset.ds.get_x_y(TRAIN_DF), dataset.ds.get_x_y(test_df)
        return (x_tr, y_tr), (x_te, y_te), cols, TRAIN_DF, test_df

    elif args.dataset == 'BONEAGE':
        return TRAIN_DF, test_df

        


################# CENSUS Dataset ####################

# US Income dataset
class CensusIncome:
    def __init__(self, path):
        self.urls = [
            "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        ]
        self.columns = [
            "age", "workClass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        self.dropped_cols = ["education", "native-country"]
        self.path = path
        self.download_dataset()
        # self.load_data(test_ratio=0.4)
        self.load_data(test_ratio=0.5)

    # Download dataset, if not present
    def download_dataset(self):
        if not os.path.exists(self.path):
            print("==> Downloading US Census Income dataset")
            os.mkdir(self.path)
            print("==> Please modify test file to remove stray dot characters")

            for url in self.urls:
                data = requests.get(url).content
                filename = os.path.join(self.path, os.path.basename(url))
                with open(filename, "wb") as file:
                    file.write(data)

    # Process, handle one-hot conversion of data etc
    def process_df(self, df):
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        def oneHotCatVars(x, colname):
            df_1 = x.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep=':')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        colnames = ['workClass', 'occupation', 'race', 'sex',
                    'marital-status', 'relationship']
        # Drop columns that do not help with task
        df = df.drop(columns=self.dropped_cols, axis=1)
        # Club categories not directly relevant for property inference
        df["race"] = df["race"].replace(
            ['Asian-Pac-Islander', 'Amer-Indian-Eskimo'], 'Other')
        for colname in colnames:
            df = oneHotCatVars(df, colname)
        # Drop features pruned via feature engineering
        prune_feature = [
            "workClass:Never-worked",
            "workClass:Without-pay",
            "occupation:Priv-house-serv",
            "occupation:Armed-Forces"
        ]
        df = df.drop(columns=prune_feature, axis=1)
        return df

    # Return data with desired property ratios
    def get_x_y(self, P):
        # Scale X values
        Y = P['income'].to_numpy()
        X = P.drop(columns='income', axis=1)
        cols = X.columns
        X = X.to_numpy()
        return (X.astype(float), np.expand_dims(Y, 1), cols)

    def get_data(self, split, prop_ratio, filter_prop, custom_limit=None, send_original_dfs=False):

        def prepare_one_set(TRAIN_DF, TEST_DF, send_original_dfs=False):
            # Apply filter to data
            TRAIN_DF = get_filter(TRAIN_DF, filter_prop,split, prop_ratio, is_test=0,custom_limit=custom_limit)
            TEST_DF = get_filter(TEST_DF, filter_prop,split, prop_ratio, is_test=1,custom_limit=custom_limit)
            # TRAIN_DF = TRAIN_DF.drop(['race:Black', 'race:Other'], axis=1)
            (x_tr, y_tr, cols), (x_te, y_te, cols) = self.get_x_y(TRAIN_DF), self.get_x_y(TEST_DF)
            if send_original_dfs:
                return (x_tr, y_tr), (x_te, y_te), cols, TRAIN_DF, TEST_DF
            return (x_tr, y_tr), (x_te, y_te), cols

        if split == "all":
            return prepare_one_set(self.train_df, self.test_df, send_original_dfs=send_original_dfs)
        if split == "prover":
            return prepare_one_set(self.train_df_victim, self.test_df_victim, send_original_dfs=send_original_dfs)
        return prepare_one_set(self.train_df_adv, self.test_df_adv, send_original_dfs=send_original_dfs)

    # Create verifier/prover splits, normalize data, etc
    def load_data(self, test_ratio, random_state=42):
        # Load train, test data
        train_data = pd.read_csv(os.path.join(self.path, 'adult.data'),names=self.columns, sep=' *, *',na_values='?', engine='python')
        test_data = pd.read_csv(os.path.join(self.path, 'adult.test'),names=self.columns, sep=' *, *', skiprows=1,na_values='?', engine='python')

        # Add field to identify train/test, process together
        train_data['is_train'] = 1
        test_data['is_train'] = 0
        df = pd.concat([train_data, test_data], axis=0)
        df = self.process_df(df)

        # Take note of columns to scale with Z-score
        z_scale_cols = ["fnlwgt", "capital-gain", "capital-loss"]
        for c in z_scale_cols:
            # z-score normalization
            df[c] = (df[c] - df[c].mean()) / df[c].std()

        # Take note of columns to scale with min-max normalization
        minmax_scale_cols = ["age",  "hours-per-week", "education-num"]
        for c in minmax_scale_cols:
            # z-score normalization
            df[c] = (df[c] - df[c].min()) / df[c].max()

        # Split back to train/test data
        self.train_df, self.test_df = df[df['is_train']== 1], df[df['is_train'] == 0]

        # Drop 'train/test' columns
        self.train_df = self.train_df.drop(columns=['is_train'], axis=1)
        self.test_df = self.test_df.drop(columns=['is_train'], axis=1)

        def s_split(this_df, rs=random_state):
            sss = StratifiedShuffleSplit(n_splits=1,test_size=test_ratio,random_state=rs)
            # Stratification on the properties we care about for this dataset so that verifier/victim split does not introduce unintended distributional shift
            splitter = sss.split(this_df, this_df[["sex:Female", "race:White", "income"]])
            split_1, split_2 = next(splitter)
            return this_df.iloc[split_1], this_df.iloc[split_2]

        # Create train/test splits for victim/verifier
        self.train_df_victim, self.train_df_adv = s_split(self.train_df)
        self.test_df_victim, self.test_df_adv = s_split(self.test_df)


def get_filter(df, filter_prop, split, ratio, is_test, custom_limit=None):
    if filter_prop == "none":
        return df
    elif filter_prop == "sex":
        def lambda_fn(x): return x['sex:Female'] == 1
    elif filter_prop == "race":
        def lambda_fn(x): return x['race:White'] == 1
    prop_wise_subsample_sizes = {
        "verifier": {
            "sex": (1100, 500),
            "race": (2000, 1000),
        },
        "prover": {
            "sex": (1100, 500),
            "race": (2000, 1000),
        },
    }

    if custom_limit is None:
        subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
    else:
        subsample_size = custom_limit
    return heuristic(df, lambda_fn, ratio,subsample_size, class_imbalance=3,n_tries=100, class_col='income',verbose=False)


class CensusWrapper:
    def __init__(self, path, filter_prop="none", ratio=0.5, split="all"):
        self.ds = CensusIncome(path)
        self.split = split
        self.ratio = ratio
        self.filter_prop = filter_prop

    def load_data(self, custom_limit=None, send_original_dfs = False):
        return self.ds.get_data(split=self.split,prop_ratio=self.ratio,filter_prop=self.filter_prop,custom_limit=custom_limit, send_original_dfs = send_original_dfs)


###################### BONEAGE ######################


class BoneDataset(Dataset):
    def __init__(self, df, argument=None, processed=False):
        if processed:
            self.features = argument
        else:
            self.transform = argument
        self.processed = processed
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.processed:
            X = self.features[self.df['path'].iloc[index]]
        else:
            # X = Image.open(self.df['path'][index])
            X = Image.open(self.df['path'][index]).convert('RGB')
            if self.transform:
                X = self.transform(X)

        y = torch.tensor(int(self.df['label'][index]))
        gender = torch.tensor(int(self.df['gender'][index]))

        return X, y, (gender)

class BoneWrapper:
    def __init__(self, df_train, df_val, features=None, augment=False):
        self.df_train = df_train
        self.df_val = df_val
        self.input_size = 224
        test_transform_list = [
            transforms.Resize((self.input_size, self.input_size)),
        ]
        train_transform_list = test_transform_list[:]

        post_transform_list = [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
        # Add image augmentations if requested
        if augment:
            train_transform_list += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(
                    shear=0.01, translate=(0.15, 0.15), degrees=5)
            ]

        train_transform = transforms.Compose(
            train_transform_list + post_transform_list)
        test_transform = transforms.Compose(
            test_transform_list + post_transform_list)

        if features is None:
            self.ds_train = BoneDataset(self.df_train, train_transform)
            self.ds_val = BoneDataset(self.df_val, test_transform)
        else:
            self.ds_train = BoneDataset(
                self.df_train, features["train"], processed=True)
            self.ds_val = BoneDataset(
                self.df_val, features["val"], processed=True)

    def get_loaders(self, batch_size, shuffle=False):
        train_loader = DataLoader(
            self.ds_train, batch_size=batch_size,
            shuffle=shuffle, num_workers=2)
        # If train mode can handle BS (weight + gradient)
        # No-grad mode can surely hadle 2 * BS?
        val_loader = DataLoader(
            self.ds_val, batch_size=batch_size * 2,
            shuffle=shuffle, num_workers=2)

        return train_loader, val_loader

# Get DF file
def get_df(path, split):
    if split not in ["prover", "verifier"]:
        raise ValueError("Invalid split specified!")

    df_train = pd.read_csv(os.path.join(path, "%s/train.csv" % split))
    df_val = pd.read_csv(os.path.join(path, "%s/val.csv" % split))

    return df_train, df_val


# Load features file
def get_features(path,split):
    if split not in ["prover", "verifier"]:
        raise ValueError("Invalid split specified!")

    # Load features
    features = {}
    features["train"] = torch.load(os.path.join(path, "%s/features_train.pt" % split))
    features["val"] = torch.load(os.path.join(path, "%s/features_val.pt" % split))

    return features


######################## Arxiv Dataset ###########################

import dgl
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

SUPPORTED_PROPERTIES = ["mean", "keep_below"]

class NodeLevelDataset:
    def __init__(self, name, normalize=True, on_gpu=True):
        self.data = DglNodePropPredDataset(name=name)

        self.g, self.labels = self.data[0]
        self.on_gpu = on_gpu

        # Extract node features
        self.features = self.g.ndata['feat']

        # Degrees of nodes
        self.degs = []

        # Normalize features
        if normalize:
            m, std = torch.mean(self.features, 0), torch.std(self.features, 0)
            self.features = (self.features - m) / std

        self.num_features = self.features.shape[1]
        self.num_classes = self.data.num_classes
        self.num_nodes = self.g.number_of_nodes()

        # Extract any extra data
        self.before_init()

        # Process graph before shifting to GPU
        self.pre_process()

        # Shift data to GPU
        if self.on_gpu:
            self.shift_to_gpu()

    def before_init(self):
        pass

    def pre_process(self):
        # Add self loops
        self.g = dgl.remove_self_loop(self.g)
        self.g = dgl.add_self_loop(self.g)

        # Make bidirectional
        self.g = dgl.to_bidirected(self.g)

    def shift_to_gpu(self):
        # Shift graph, labels to cuda
        self.g = self.g.to(device)
        self.labels = self.labels.to(device)
        self.features = self.features.to(device)

    def get_idx_split(self, test_ratio=0.2):
        num_test = int(test_ratio * self.num_nodes)
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        train_mask[num_test:] = 1
        test_mask = torch.zeros(
            self.num_nodes, dtype=torch.bool)
        test_mask[:num_test] = 1
        return train_mask, test_mask

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def precompute_degrees(self):
        X, _ = self.g.edges()
        degs = []
        for i in tqdm(range(self.g.number_of_nodes())):
            degs.append(torch.sum(X == i).item())
        self.degs = np.array(degs)


class ArxivNodeDataset(NodeLevelDataset):
    def __init__(self, split, normalize=True, on_gpu=True):
        super(ArxivNodeDataset, self).__init__(
            'ogbn-arxiv', normalize=normalize, on_gpu=on_gpu)

        # 59:41 victim:verifier data split
        # (all original data, including train/val/test)
        # Original data had 54:46 train-nontrain split
        # Get similar splits
        split_year = 2016
        if split == 'verifier':
            # 77:23 train:test split
            test_year = 2015
            self.train_idx = self.years < test_year
            self.test_idx = torch.logical_and(
                self.years >= test_year, self.years < split_year)
        elif split == 'prover':
            # 66:34 train:test split
            test_year = 2019
            self.train_idx = torch.logical_and(
                self.years != test_year, self.years >= split_year)
            self.test_idx = (self.years == test_year)
        else:
            raise ValueError("Invalid split requested!")

        self.train_idx = torch.nonzero(self.train_idx, as_tuple=True)[0]
        self.test_idx = torch.nonzero(self.test_idx, as_tuple=True)[0]

        # Sort them now, for easier access later
        self.train_idx = torch.sort(self.train_idx)[0]
        self.test_idx = torch.sort(self.test_idx)[0]

    def before_init(self):
        # Extract years
        self.years = torch.squeeze(self.g.ndata['year'], 1)
        if self.on_gpu:
            self.years = self.years.to(device)

    def get_idx_split(self):
        return self.train_idx, self.test_idx

    def random_split_pick(self, ratio):
        # Randomly sample 'ratio' worth of specified train data
        # Set train mask to those ones
        n_elems = len(self.train_idx)
        perm = torch.randperm(n_elems)[:int(ratio * n_elems)]
        self.train_idx = self.train_idx[perm]

    def label_ratio_preserving_pick(self, total_tr, total_te):
        # While maintaining relative label ratios for classes
        # Sample a set of size total
        labels = torch.cat(
            (self.labels[self.train_idx, 0], self.labels[self.test_idx, 0]), 0)
        elems, counts = torch.unique(labels, return_counts=True)
        # Get ratios for these elems
        counts = counts.float() / len(labels)
        # Sample according to ratio from existing train, test sets
        train_new, test_new = [], []
        for e, c in zip(elems, counts):
            # for train
            qualify = torch.nonzero(
                self.labels[self.train_idx, 0] == e, as_tuple=True)[0]
            ids = torch.randperm(len(qualify))[:int(c * total_tr)]
            train_new.append(qualify[ids])
            # for test
            qualify = torch.nonzero(
                self.labels[self.test_idx, 0] == e, as_tuple=True)[0]
            ids = torch.randperm(len(qualify))[:int(c * total_te)]
            test_new.append(qualify[ids])

        self.train_idx = torch.cat(train_new, 0)
        self.test_idx = torch.cat(test_new, 0)

    def change_property(self, wanted_degree, change, prune_ratio=0.0):
        '''
        Prune graph nodes such that mean degree of remaining
        graph is the same as requested degree.
        '''
        # If no change requested, perform no change
        if wanted_degree is None:
            return

        # Compute degrees
        self.precompute_degrees()

        # Prune graph, get rid of nodes
        label_ids = torch.cat((self.train_idx, self.test_idx)).cpu().numpy()
        self.g, pruned_nodes = change_graph_property(
            self.g, self.degs, wanted_degree, change,
            label_ids, prune_ratio=prune_ratio)

        # Make mapping between old and new IDs
        not_pruned = torch.ones(self.num_nodes).byte()
        not_pruned[pruned_nodes] = False
        not_pruned = torch.nonzero(not_pruned, as_tuple=True)[0]
        mapping = {x.item(): i for i, x in enumerate(not_pruned)}

        # Function to modify current masks to reflect pruning
        def process(ids):
            # Convert indices to mask
            keep = torch.zeros(self.num_nodes).byte()
            keep[ids] = True

            # Get rid of pruned nodes from this, get back to IDs
            not_pruned = torch.ones(self.num_nodes).byte()
            not_pruned[pruned_nodes] = False
            not_pruned = torch.nonzero(not_pruned, as_tuple=True)[0]
            keep[pruned_nodes] = False
            keep = torch.nonzero(keep, as_tuple=True)[0]

            # Update current mask to point to correct IDs
            for i, x in enumerate(keep):
                keep[i] = mapping[x.item()]

            # Shift mask back to GPU
            if self.on_gpu:
                keep = keep.to(device)
            return keep

        # Update masks to account for pruned nodes,  re-indexing
        self.train_idx = process(self.train_idx)
        self.test_idx = process(self.test_idx)

        # Update features, labels, year-information
        self.years = self.years[not_pruned]
        self.features = self.features[not_pruned]
        self.labels = self.labels[not_pruned]

        # Update number of nodes in graph
        self.num_nodes -= len(pruned_nodes)

    def change_mean_degree(self, wanted_degree, prune_ratio=0.0):
        '''
        Prune graph nodes such that mean degree of remaining
        graph is the same as requested degree.
        '''
        self.change_property(wanted_degree, "mean", prune_ratio)

    def keep_below_degree_threshold(self, threshold_degree, prune_ratio=0.0):
        '''
        Prune graph nodes such that remaining nodes all have
        node degree strictly below threshold
        '''
        self.change_property(threshold_degree, "keep_below", prune_ratio)


def neighbor_removal(g, pick, degs, inf_val):
    L, R = g.edges()
    neighbors = R[L == pick].cpu().numpy()
    neighbors_mask = np.zeros_like(degs, dtype=bool)
    neighbors_mask[neighbors] = True
    neighbors_mask[degs == inf_val] = False
    degs[neighbors_mask] -= 1
    return degs


def find_to_prune_for_mean_degree(g, degs, wanted_deg, pre_prune=[]):
    prune = []
    degs = degs.astype(np.float32)

    # Define logic to process removal of node
    def loop_step(pick):
        nonlocal degs, prune

        # Reduce degree of all nodes that were connected to pruned node
        # And are up for consideration currently
        degs = neighbor_removal(g, pick, degs, np.nan)

        # Removed this node, mark as NAN
        degs[pick] = np.nan

        # Note node to be pruned and keep track of current degree
        cur_deg = np.nanmean(degs)
        prune.append(pick)

        return cur_deg

    # If nodes to be pruned already provided, add them to list
    # and be mindful of their pruning
    for pp in pre_prune:
        cur_deg = loop_step(pp)

    # Take note of mean degree right now
    cur_deg = np.nanmean(degs)

    # If desired degree is more than current, prune low-degree nodes
    if wanted_deg > cur_deg:
        while cur_deg < wanted_deg:
            # Find minimum right now
            pick = np.nanargmin(degs)

            # Process removal of this node
            cur_deg = loop_step(pick)

    # Else, prune high-degree nodes
    else:
        while cur_deg > wanted_deg:
            # Find minimum right now
            pick = np.nanargmax(degs)

            # Process removal of this node
            cur_deg = loop_step(pick)

    # Return list of nodes that should be removed
    return prune


def find_to_prune_for_threshold_degree(
    g, degs, threshold_degree, pre_prune=[]):
    prune = []
    inf_val = -1

    # Define logic to process removal of node
    def loop_step(pick):
        nonlocal degs, prune

        # Note node to be pruned
        prune.append(pick)

        # Reduce degree of all nodes that were connected to pruned node
        # And are up for consideration currently
        degs = neighbor_removal(g, pick, degs, inf_val)

        # Removed this node, mark as -1
        degs[pick] = inf_val

        return degs

    # If nodes to be pruned already provided, add them to list
    # and be mindful of their pruning
    for pp in pre_prune:
        degs = loop_step(pp)

    # Keep going until maximum node degree in graph is below threshold
    pick = np.argmax(degs)
    while degs[pick] >= threshold_degree:

        # Process removal of this node
        degs = loop_step(pick)

        # Find maximum degree node right now
        pick = np.argmax(degs)

    # Make sure other nodes have strictly smaller degree
    assert np.all(degs < threshold_degree)

    return prune


def change_graph_property(
        g, degs, wanted_deg, change,
        label_ids, prune_ratio=0):

    # Randomly pick and prune non-label nodes for graph
    # To introduce some randomness in graph structure
    pre_prune = []

    if prune_ratio > 0:
        # Since distribution according to connectivity is uneven
        # Bin according to degree to avoid pruning only
        # Low degree nodes

        bins = [0, 5, 10, 20, 50, np.inf]
        for i in range(len(bins)-1):
            picked = np.logical_and(degs >= bins[i], degs < bins[i+1])
            picked = picked.nonzero()[0]
            n_wanted = int(prune_ratio * len(picked))

            # Avoid nodes with labels
            picked = np.setdiff1d(picked, label_ids, assume_unique=True)

            perm = np.random.permutation(
                len(picked))[:n_wanted]
            pre_prune += list(perm)

    if change not in SUPPORTED_PROPERTIES:
        raise NotImplementedError("Not implemented this property")

    if change == "mean":
        to_prune = find_to_prune_for_mean_degree(
            g, degs, wanted_deg, pre_prune)
    else:
        to_prune = find_to_prune_for_threshold_degree(
            g, degs, wanted_deg, pre_prune)

    # Get rid of these nodes from graph
    g = dgl.remove_nodes(g, to_prune)

    return g, to_prune
