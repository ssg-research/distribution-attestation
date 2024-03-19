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

import torch
import torch.nn as nn


class BinaryNet(torch.nn.Module):
    def __init__(self, num_features):
        super(BinaryNet, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, 1024),
            nn.Tanh(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            nn.Tanh(),
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            nn.Tanh(),
        )
        self.fc4 = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.classifier = torch.nn.Linear(128, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)


# class BinaryNet(torch.nn.Module):
#     def __init__(self, num_features):
#         super(BinaryNet, self).__init__()
#         self.fc1 = torch.nn.Sequential(
#             torch.nn.Linear(num_features, 1024),
#         )
#         self.fc2 = torch.nn.Sequential(
#             torch.nn.Linear(1024, 512),
#         )
#         self.fc3 = torch.nn.Sequential(
#             torch.nn.Linear(512, 256),
#         )
#         self.fc4 = torch.nn.Sequential(
#             torch.nn.Linear(256, 128),
#         )
#         self.classifier = torch.nn.Linear(128, 1)
#
#     def forward(self, x):
#         out = nn.functional.tanh(self.fc1(x))
#         out = nn.functional.tanh(self.fc2(out))
#         out = nn.functional.tanh(self.fc3(out))
#         out = nn.functional.tanh(self.fc4(out))
#         return self.classifier(out)