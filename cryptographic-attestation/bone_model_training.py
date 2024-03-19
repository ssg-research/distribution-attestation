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

import random
import time
import argparse
import os

import crypten
import crypten.nn
import crypten.optim
import crypten.communicator as comm

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models import bone_model
from multiprocess_launcher import MultiProcessLauncher
from utils import get_device, AverageMeter


###################### BONEAGE dataset and model ######################

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

def get_features(path,split):
    if split not in ["prover", "verifier"]:
        raise ValueError("Invalid split specified!")

    # Load features
    features = {}
    features["train"] = torch.load(os.path.join(path, "%s/features_train.pt" % split))
    features["val"] = torch.load(os.path.join(path, "%s/features_val.pt" % split))

    return features


def dataloader_to_tensors(bone_ds):
    df_list = []
    labels_list = []
    # for data in dl
    for i in range(len(bone_ds)):
        images, labels, _ = bone_ds[i]
        df_list.append(images)
        labels_list.append(labels)
    data_tensor = torch.stack(df_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.float)
    return data_tensor, labels_tensor


def load_data(path):
    # load train data
    train = torch.load(path + "features_train.pt")

    train_df = pd.read_csv(path + "train.csv")
    train_data = torch.stack([train[name] for name in train_df["path"]])
    train_labels = torch.tensor(train_df["label"], dtype=torch.float)
    train_gender = torch.tensor(train_df["gender"], dtype=torch.float)

    # load validation data
    val = torch.load(path + "features_val.pt")

    val_df = pd.read_csv(path + "val.csv")
    val_data = torch.stack([val[name] for name in val_df["path"]])
    val_labels = torch.tensor(val_df["label"], dtype=torch.float)

    return train_data, train_labels, train_gender, val_data, val_labels


def experiment(args):
    device = get_device(device=args.gpu)
    features = 1024

    args.rank = comm.get().get_rank() if args.crypten else 0

    # set RNG seeds
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    # load verifier data
    train_data, train_labels, train_gender, val_data, val_labels = load_data("./data/boneage_splits/verifier/")
    train_data, train_labels, train_gender, val_data, val_labels = train_data.to(device), train_labels.to(device), train_gender.to(device), val_data.to(device), val_labels.to(device)
    num_samples = train_data.size(0)
    num_val_samples = val_data.size(0)

    attestation_value = args.attestation_value

    # setup training
    if args.crypten:
        dummy_input = torch.empty(size=(args.batch_size, features))
        model = crypten.nn.from_pytorch(bone_model.BoneModel(n_inp=features), dummy_input).to(device).encrypt()
        criterion = crypten.nn.BCEWithLogitsLoss()
        optimizer = crypten.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

        # secrete share data
        train_data, train_labels, val_data = crypten.cryptensor(train_data), crypten.cryptensor(train_labels), crypten.cryptensor(val_data)
        train_gender = crypten.cryptensor(train_gender)

        # setup attestation
        one_enc = crypten.cryptensor(torch.ones(train_gender.size())).to(device)
        zero_enc = crypten.cryptensor(torch.zeros(train_gender.size())).to(device)
        attestation_value = crypten.cryptensor(torch.tensor([attestation_value])).to(device)

    else:
        model = bone_model.BoneModel(n_inp=features).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

    if args.crypten:
        comm.get().reset_communication_stats()
    start_time = time.time()

    # Attestation
    if args.crypten:
        attestation = crypten.where(train_gender == attestation_value, one_enc, zero_enc)
        distributional_property = attestation.sum() / train_gender.size(0)
        distributional_property_plaintext = distributional_property.get_plain_text()
        distributional_property_satisfied = distributional_property_plaintext.item() == args.distribution
    else:
        attestation = train_gender == attestation_value
        distributional_property = attestation.sum() / train_gender.size(0)
        distributional_property_plaintext = distributional_property
        distributional_property_satisfied = distributional_property_plaintext.item() == args.distribution

    if args.rank == 0:
        print(f"Distribution of Gender with label {args.attestation_value}: {distributional_property_plaintext}")
        print(f"Passed Distributional Property Check: {distributional_property_satisfied}")

    end_time = time.time()
    if args.crypten:
        comm.get().print_communication_stats()
        communication = comm.get().get_communication_stats()

    if args.rank == 0:
        print("\nDistributional Property Check Results")
        print(f"Run Time: {end_time - start_time}")
        if args.crypten:
            print(f"Communication: {communication}")

    if args.crypten:
        comm.get().reset_communication_stats()
    start_time = time.time()

    # Training
    if args.rank == 0:
        print("\nTraining")

    val_acc = AverageMeter()
    for e in range(args.epochs):
        if args.rank == 0:
            print(f"Epoch: {e}")

        # training
        for i in range(0, num_samples, args.batch_size):
            start, end = i, min(i + args.batch_size, num_samples)
            # print(i)

            inputs = train_data[start:end]
            inputs.requires_grad = True
            labels = train_labels[start:end].reshape(-1, 1)
            labels.requires_grad = True

            # if args.crypten:
            #     print(inputs.get_plain_text())
            #     print(labels.get_plain_text())

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if args.debug:
                if args.crypten:
                    plain_loss = loss.get_plain_text()
                    if args.rank == 0:
                        print(f'loss: {plain_loss}')
                else:
                    print(f'loss: {loss}')

            loss.backward()
            optimizer.step()


    end_time = time.time()
    if args.crypten:
        comm.get().print_communication_stats()
        communication = comm.get().get_communication_stats()

    if args.rank == 0:
        print("Training")
        print(f"Run Time: {end_time - start_time}")
        if args.crypten:
            print(f"Communication: {communication}")


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser()

    ### training settings
    parser.add_argument("--batch_size", help="batch size", type=int, default=256)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--momentum", help="momentum", type=float, default=0)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.0001)
    parser.add_argument("--epochs", help="# iterations", type=int, default=100)
    parser.add_argument("--gpu", help="no gpu = -1, gpu training otherwise", type=int, default=0)
    parser.add_argument("--seed", help="seed for randomness", type=int, default=1)
    parser.add_argument("--distribution", help="distribution training data should follow", type=float, default=0.5)
    parser.add_argument("--attestation_value", help="value to check distribution", type=float, default=1)
    parser.add_argument("--debug", help="train with multiple processes", action="store_true", default=True)

    # CrypTen exclusive
    parser.add_argument("--crypten", help="train with multiple processes", action="store_true", default=False)
    parser.add_argument("--multiprocess", help="train with multiple parties on a single GPU", action="store_true", default=False)
    parser.add_argument("--distributed", help="train with multiple parties on a multiple GPU", action="store_true", default=False)
    parser.add_argument("--world_size", type=int, default=2, help="The number of parties to launch. Each party acts as its own process")

    args = parser.parse_args()

    if args.crypten and args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    elif args.crypten and args.distributed:
        device = get_device(args.gpu)
        crypten.init(config_file="./crypten_config/multiprocess_config.yaml", device=device)
        experiment(args)
    else:
        if args.crypten:
            device = get_device(args.gpu)
            crypten.init(config_file="./crypten_config/singleprocess_config.yaml", device=device)
        experiment(args)
