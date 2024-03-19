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

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 9 > ARXIV_bb_9_adv_vuln_e8255.log

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 10 > ARXIV_bb_10_adv_vuln_e8255.log

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 11 > ARXIV_bb_11_adv_vuln_e8255.log

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 12 > ARXIV_bb_12_adv_vuln_e8255.log

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 13 > ARXIV_bb_13_adv_vuln_e8255.log

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 14 > ARXIV_bb_14_adv_vuln_e8255.log

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 15 > ARXIV_bb_15_adv_vuln_e8255.log

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 16 > ARXIV_bb_16_adv_vuln_e8255.log

python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 17 > ARXIV_bb_17_adv_vuln_e8255.log



# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 9 > ARXIV_bb_9_adv_gen_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 10 > ARXIV_bb_10_adv_gen_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 11 > ARXIV_bb_11_adv_gen_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 12 > ARXIV_bb_12_adv_gen_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 13 > ARXIV_bb_13_adv_gen_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 14 > ARXIV_bb_14_adv_gen_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 15 > ARXIV_bb_15_adv_gen_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 16 > ARXIV_bb_16_adv_gen_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_defense  --claim_degree 17 > ARXIV_bb_17_adv_gen_e8255.log



# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 9 > ARXIV_bb_9_adv_defended_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 10 > ARXIV_bb_10_adv_defended_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 11 > ARXIV_bb_11_adv_defended_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 12 > ARXIV_bb_12_adv_defended_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 13 > ARXIV_bb_13_adv_defended_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 14 > ARXIV_bb_14_adv_defended_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 15 > ARXIV_bb_15_adv_defended_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 16 > ARXIV_bb_16_adv_defended_e8255.log

# python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 17 > ARXIV_bb_17_adv_defended_e8255.log