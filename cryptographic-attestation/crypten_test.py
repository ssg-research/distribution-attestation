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

import crypten
import torch


if __name__ == "__main__":
    crypten.init(config_file="./crypten_config/singleprocess_config.yaml")

    a = torch.ones((10, 10))
    a_enc = crypten.cryptensor(a)

    print(f"PyTorch Tensor: {a}")
    print(f"Encrypted CrypTen Tensor: {a_enc}")
    print(f"Revealed CrypTen Tensor: {a_enc.get_plain_text()}")