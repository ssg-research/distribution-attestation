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
# used as: bone_model_nn = models.BoneModel(n_inp=1024)

"""
Another representation of BoneModel was:
BoneModel(
  (layers): Sequential(
    (0): Linear(in_features=1024, out_features=128, bias=True)
    (1): FakeReluWrapper()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): FakeReluWrapper()
    (4): Linear(in_features=64, out_features=1, bias=True)
  )
)
"""

class BoneModel(nn.Module):
    def __init__(self, n_inp: int, fake_relu: bool = False, latent_focus: int = None):
        if latent_focus is not None:
            if latent_focus not in [0, 1]:
                raise ValueError("Invalid interal layer requested")

        if fake_relu:
            #act_fn = BasicWrapper
            pass
        else:
            act_fn = nn.ReLU

        self.latent_focus = latent_focus
        super(BoneModel, self).__init__()
        # just ignore the wrapper and treat as ReLU
        #layers = [nn.Linear(n_inp, 128), FakeReluWrapper(inplace=True), nn.Linear(128, 64), FakeReluWrapper(inplace=True), nn.Linear(64, 1)]
        layers = [nn.Linear(n_inp, 128), nn.ReLU(inplace=True), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)]

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

if __name__ == "__main__":
    model = BoneModel(n_inp=1024)

    x1 = torch.rand(size=(1, 1024))
    x2 = torch.rand(size=(4, 1024))

    y1 = model(x1)
    y2 = model(x2)

    print(y1.size, y2.size)
