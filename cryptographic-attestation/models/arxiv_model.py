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

from dgl.nn.pytorch import GraphConv
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

import crypten

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class GCN(nn.Module):
    def __init__(self, ds, n_hidden, n_layers, dropout):
        super(GCN, self).__init__()
        # Get actual graph
        self.g = ds.g

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(ds.num_features, n_hidden, activation=F.relu))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.relu))

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


# For training
num_epochs_to_use = 500
lr_to_use = 1e-4

if __name__ == "__main__":
    crypten.init(config_file="../crypten_config/singleprocess_config.yaml", device=device)

    model = GraphConv(100, 1000, activation=F.relu)

    a = torch.empty(size=(100, 100))

    b = model(a)

    print(model)