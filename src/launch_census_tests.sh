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

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.0 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_00_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.1 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_01_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.2 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_02_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.3 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_03_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.4 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_04_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.5 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_05_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.6 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_06_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.7 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_07_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.8 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_08_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.9 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_09_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 1.0 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_s_10_test_adv_train.log


# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.0 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_00_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.1 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_01_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.2 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_02_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.3 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_03_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.4 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_04_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.5 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_05_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.6 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_06_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.7 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_07_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.8 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_08_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.9 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_09_test_adv_train.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 1.0 --use_single_attest --do_defense --classifier_save_suffix "_defended" >> src/CENSUS_r_10_test_adv_train.log





python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.0 --use_single_attest >> src/CENSUS_s_00_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.1 --use_single_attest >> src/CENSUS_s_01_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.2 --use_single_attest >> src/CENSUS_s_02_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.3 --use_single_attest >> src/CENSUS_s_03_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.4 --use_single_attest >> src/CENSUS_s_04_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.5 --use_single_attest >> src/CENSUS_s_05_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.6 --use_single_attest >> src/CENSUS_s_06_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.7 --use_single_attest >> src/CENSUS_s_07_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.8 --use_single_attest >> src/CENSUS_s_08_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.9 --use_single_attest >> src/CENSUS_s_09_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 1.0 --use_single_attest >> src/CENSUS_s_10_test_single.log


python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.0 --use_single_attest >> src/CENSUS_r_00_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.1 --use_single_attest >> src/CENSUS_r_01_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.2 --use_single_attest >> src/CENSUS_r_02_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.3 --use_single_attest >> src/CENSUS_r_03_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.4 --use_single_attest >> src/CENSUS_r_04_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.5 --use_single_attest >> src/CENSUS_r_05_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.6 --use_single_attest >> src/CENSUS_r_06_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.7 --use_single_attest >> src/CENSUS_r_07_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.8 --use_single_attest >> src/CENSUS_r_08_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.9 --use_single_attest >> src/CENSUS_r_09_test_single.log

python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 1.0 --use_single_attest >> src/CENSUS_r_10_test_single.log





# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.0 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_00_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.1 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_01_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.2 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_02_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.3 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_03_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.4 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_04_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.5 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_05_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.6 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_06_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.7 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_07_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.8 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_08_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 0.9 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_09_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter sex --claim_ratio 1.0 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_s_10_test_single_prover.log


# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.0 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_00_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.1 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_01_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.2 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_02_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.3 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_03_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.4 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_04_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.5 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_05_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.6 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_06_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.7 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_07_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.8 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_08_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 0.9 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_09_test_single_prover.log

# python -m src.train_verifier_model --dataset CENSUS --filter race --claim_ratio 1.0 --use_single_attest --train_prover_version --classifier_save_suffix "_prover" >> src/CENSUS_r_10_test_single_prover.log