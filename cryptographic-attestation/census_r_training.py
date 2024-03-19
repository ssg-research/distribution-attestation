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

import crypten
import crypten.nn
import crypten.optim
import crypten.communicator as comm

import torch
import numpy as np
import pandas as pd

from models import census_model
from multiprocess_launcher import MultiProcessLauncher
from utils import get_device, AverageMeter




def data_loading(num_samples, num_features, num_labels):
    # all categorical features are represented by integers
    train_data = torch.stack([
        torch.randint(18, 100, size=(num_samples, )),   # age
        torch.randint(0, 8, size=(num_samples, )),   # workclass
        torch.randint(1000, 1000000, size=(num_samples, )),   # fnlwgt
        torch.randint(0, 16, size=(num_samples, )),   # education
        torch.randint(0, 16, size=(num_samples, )),   # education-num
        torch.randint(0, 7, size=(num_samples, )),   # marital-status
        torch.randint(0, 14, size=(num_samples, )),   # occupation
        torch.randint(0, 6, size=(num_samples, )),   # relationship
        torch.randint(0, 5, size=(num_samples, )),   # race
        torch.randint(0, 2, size=(num_samples, )),   # sex
        torch.randint(0, 100000, size=(num_samples, )),   # capital-gain
        torch.randint(0, 100000, size=(num_samples, )),   # capital-loss
        torch.randint(0, 60, size=(num_samples, )),   # hours-per-week
        torch.randint(0, 43, size=(num_samples, )),   # native-country
    ]).T

    train_data = train_data.to(torch.float)
    train_labels = torch.randint(0, num_labels, size=(num_samples, )).to(torch.float)

    return train_data, train_labels


def experiment(args):
    device = get_device(device=args.gpu)
    features = 14

    args.rank = comm.get().get_rank() if args.crypten else 0

    # set RNG seeds
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    # load "data"
    train_data, train_labels = data_loading(1800, features, 2)   # load random data of the right shape
    train_data, train_labels = train_data.to(device), train_labels.to(device)
    num_samples = train_data.size(0)

    attestation_value = args.attestation_value

    if args.crypten:
        dummy_input = torch.empty(size=(args.batch_size, features))
        model = crypten.nn.from_pytorch(census_model.BinaryNet(num_features=features), dummy_input).to(device).encrypt()
        criterion = crypten.nn.BCEWithLogitsLoss()
        optimizer = crypten.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

        train_data, train_labels = crypten.cryptensor(train_data), crypten.cryptensor(train_labels)

        # setup attestation
        one_enc = crypten.cryptensor(torch.ones(num_samples)).to(device)
        zero_enc = crypten.cryptensor(torch.zeros(num_samples)).to(device)
        attestation_value = crypten.cryptensor(torch.tensor([attestation_value])).to(device)

    else:
        model = census_model.BinaryNet(num_features=features).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

    if args.crypten:
        comm.get().reset_communication_stats()
    start_time = time.time()

    # Attestation
    feature_idx = 8

    if args.crypten:
        attestation = crypten.where(train_data[:, feature_idx] == attestation_value, one_enc, zero_enc)
        distributional_property = attestation.sum() / num_samples
        distributional_property_plaintext = distributional_property.get_plain_text()
        distributional_property_satisfied = distributional_property_plaintext.item() == args.distribution
    else:
        attestation = train_data[:, feature_idx] == attestation_value
        distributional_property = attestation.sum() / num_samples
        distributional_property_plaintext = distributional_property
        distributional_property_satisfied = distributional_property_plaintext.item() == args.distribution

    if args.rank == 0:
        print(f"Distribution of Feature {feature_idx} with label {args.attestation_value}: {distributional_property_plaintext}")
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
    test_data, test_labels = data_loading(1600, features, 2)   # load random data of the right shape
    test_data, test_labels = test_data.to(device), test_labels.to(device)
    if args.crypten:
        test_data = crypten.cryptensor(test_data)

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
        print("\nTraining Results")
        print(f"Run Time: {end_time - start_time}")
        if args.crypten:
            print(f"Communication: {communication}")


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser()

    ### training settings
    parser.add_argument("--batch_size", help="batch size", type=int, default=256)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--momentum", help="momentum", type=float, default=0)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.0001)
    parser.add_argument("--epochs", help="# iterations", type=int, default=30)
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
