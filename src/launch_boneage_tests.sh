#!/bin/bash

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

# python -m src.train_verifier_model --dataset BONEAGE --do_defense --claim_ratio 0.2 --classifier_save_suffix "_defended" >> src/BONEAGE_02_adv_train.log

# python -m src.train_verifier_model --dataset BONEAGE --do_defense --claim_ratio 0.3 --classifier_save_suffix "_defended" >> src/BONEAGE_03_adv_train.log

# python -m src.train_verifier_model --dataset BONEAGE --do_defense --claim_ratio 0.4 --classifier_save_suffix "_defended" >> src/BONEAGE_04_adv_train.log

# python -m src.train_verifier_model --dataset BONEAGE --do_defense --claim_ratio 0.5 --classifier_save_suffix "_defended" >> src/BONEAGE_05_adv_train.log

# python -m src.train_verifier_model --dataset BONEAGE --do_defense --claim_ratio 0.6 --classifier_save_suffix "_defended" >> src/BONEAGE_06_adv_train.log

# python -m src.train_verifier_model --dataset BONEAGE --do_defense --claim_ratio 0.7 --classifier_save_suffix "_defended" >> src/BONEAGE_07_adv_train.log

# python -m src.train_verifier_model --dataset BONEAGE --do_defense --claim_ratio 0.8 --classifier_save_suffix "_defended" >> src/BONEAGE_08_adv_train.log


python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.2 >> src/BONEAGE_02_test.log

python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.3 >> src/BONEAGE_03_test.log

python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.4 >> src/BONEAGE_04_test.log

python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.5 >> src/BONEAGE_05_test.log

python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.6 >> src/BONEAGE_06_test.log

python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.7 >> src/BONEAGE_07_test.log

python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.8 >> src/BONEAGE_08_test.log



# python -m src.train_verifier_model --dataset BONEAGE  --claim_ratio 0.2 --train_prover_version --classifier_save_suffix "_prover" >> src/BONEAGE_02_test_prover.log

# python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.3 --train_prover_version --classifier_save_suffix "_prover" >> src/BONEAGE_03_test_prover.log

# python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.4 --train_prover_version --classifier_save_suffix "_prover" >> src/BONEAGE_04_test_prover.log

# python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.5 --train_prover_version --classifier_save_suffix "_prover" >> src/BONEAGE_05_test_prover.log

# python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.6 --train_prover_version --classifier_save_suffix "_prover" >> src/BONEAGE_06_test_prover.log

# python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.7 --train_prover_version --classifier_save_suffix "_prover" >> src/BONEAGE_07_test_prover.log

# python -m src.train_verifier_model --dataset BONEAGE --claim_ratio 0.8 --train_prover_version --classifier_save_suffix "_prover" >> src/BONEAGE_08_test_prover.log