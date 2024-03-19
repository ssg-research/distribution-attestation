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

This file will list the hyperparameters for all of the datasets.

###CENSUS
('--lr', type=float, default=0.01)
('--batch_size', type=int, default=64)
('--decay', type = float, default = 0, help = "Weight decay/L2 Regularization")
('--epochs', type=int, default=100)


###BONEAGE
lr=1e-3,
epoch_num=100,
batch_size = 256*32,
weight_decay=1e-4


###ARXIV
prune = 0
lr = 0.01
epochs = 100
weight_decay = 5e-4
dropout = 0.5
