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

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 9 --classifier_save_suffix "_defended" >> src/ARXIV_9_adv_train.log

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 10 --classifier_save_suffix "_defended" >> src/ARXIV_10_adv_train.log

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 11 --classifier_save_suffix "_defended" >> src/ARXIV_11_adv_train.log

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 12 --classifier_save_suffix "_defended" >> src/ARXIV_12_adv_train.log

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 13 --classifier_save_suffix "_defended" >> src/ARXIV_13_adv_train.log

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 14 --classifier_save_suffix "_defended" >> src/ARXIV_14_adv_train.log

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 15 --classifier_save_suffix "_defended" >> src/ARXIV_15_adv_train.log

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 16 --classifier_save_suffix "_defended" >> src/ARXIV_16_adv_train.log

# python -m src.train_verifier_model --dataset ARXIV --do_defense --claim_degree 17 --classifier_save_suffix "_defended" >> src/ARXIV_17_adv_train.log


python -m src.train_verifier_model --dataset ARXIV --claim_degree 9 >> src/ARXIV_9_test.log

python -m src.train_verifier_model --dataset ARXIV --claim_degree 10 >> src/ARXIV_10_test.log

python -m src.train_verifier_model --dataset ARXIV --claim_degree 11 >> src/ARXIV_11_test.log

python -m src.train_verifier_model --dataset ARXIV --claim_degree 12 >> src/ARXIV_12_test.log

python -m src.train_verifier_model --dataset ARXIV --claim_degree 13 >> src/ARXIV_13_test.log

python -m src.train_verifier_model --dataset ARXIV --claim_degree 14 >> src/ARXIV_14_test.log

python -m src.train_verifier_model --dataset ARXIV --claim_degree 15 >> src/ARXIV_15_test.log

python -m src.train_verifier_model --dataset ARXIV --claim_degree 16 >> src/ARXIV_16_test.log

python -m src.train_verifier_model --dataset ARXIV --claim_degree 17 >> src/ARXIV_17_test.log


# python -m src.train_verifier_model --dataset ARXIV --claim_degree 9 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_9_test_prover.log

# python -m src.train_verifier_model --dataset ARXIV --claim_degree 10 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_10_test_prover.log

# python -m src.train_verifier_model --dataset ARXIV --claim_degree 11 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_11_test_prover.log

# python -m src.train_verifier_model --dataset ARXIV --claim_degree 12 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_12_test_prover.log

# python -m src.train_verifier_model --dataset ARXIV --claim_degree 13 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_13_test_prover.log

# python -m src.train_verifier_model --dataset ARXIV --claim_degree 14 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_14_test_prover.log

# python -m src.train_verifier_model --dataset ARXIV --claim_degree 15 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_15_test_prover.log

# python -m src.train_verifier_model --dataset ARXIV --claim_degree 16 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_16_test_prover.log

# python -m src.train_verifier_model --dataset ARXIV --claim_degree 17 --train_prover_version --classifier_save_suffix "_prover" >> src/ARXIV_17_test_prover.log
