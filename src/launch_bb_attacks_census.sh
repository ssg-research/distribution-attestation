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


# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.0 > CENSUS_s_bb_0.0_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.1 > CENSUS_s_bb_0.1_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.2 > CENSUS_s_bb_0.2_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.3 > CENSUS_s_bb_0.3_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.4 > CENSUS_s_bb_0.4_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.5 > CENSUS_s_bb_0.5_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.6 > CENSUS_s_bb_0.6_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.7 > CENSUS_s_bb_0.7_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.8 > CENSUS_s_bb_0.8_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 0.9 > CENSUS_s_bb_0.9_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_defense  --claim_ratio 1.0 > CENSUS_s_bb_1.0_adv_gen.log


# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 0.0 > CENSUS_r_bb_0.0_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _r_01 --filter race --do_defense  --claim_ratio 0.1 > CENSUS_r_bb_0.1_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 0.2 > CENSUS_r_bb_0.2_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 0.3 > CENSUS_r_bb_0.3_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 0.4 > CENSUS_r_bb_0.4_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 0.5 > CENSUS_r_bb_0.5_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 0.6 > CENSUS_r_bb_0.6_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter race --do_defense  --claim_ratio 0.7 > CENSUS_r_bb_0.7_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 0.8 > CENSUS_r_bb_0.8_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 0.9 > CENSUS_r_bb_0.9_adv_gen.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_defense  --claim_ratio 1.0 > CENSUS_r_bb_1.0_adv_gen.log


# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.0 > CENSUS_s_bb_0.0_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.1 > CENSUS_s_bb_0.1_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.2 > CENSUS_s_bb_0.2_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.3 > CENSUS_s_bb_0.3_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.4 > CENSUS_s_bb_0.4_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.5 > CENSUS_s_bb_0.5_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.6 > CENSUS_s_bb_0.6_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.7 > CENSUS_s_bb_0.7_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.8 > CENSUS_s_bb_0.8_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.9 > CENSUS_s_bb_0.9_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 1.0 > CENSUS_s_bb_1.0_adv_vuln.log


# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.0 > CENSUS_r_bb_0.0_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _r_01 --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover_r_01 --claim_ratio 0.1 > CENSUS_r_bb_0.1_adv_vuln2.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.2 > CENSUS_r_bb_0.2_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.3 > CENSUS_r_bb_0.3_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.4 > CENSUS_r_bb_0.4_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix Feb23Ideal --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.5 > CENSUS_r_bb_0.5_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.6 > CENSUS_r_bb_0.6_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.7 > CENSUS_r_bb_0.7_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.8 > CENSUS_r_bb_0.8_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.9 > CENSUS_r_bb_0.9_adv_vuln.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_vuln --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 1.0 > CENSUS_r_bb_1.0_adv_vuln.log



# python -m src.verification --dataset CENSUS --classifier_suffix _defended_s --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.0 > CENSUS_s_bb_0.0_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_s --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.1 > CENSUS_s_bb_0.1_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_s --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.2 > CENSUS_s_bb_0.2_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_adv --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --test_defended --claim_ratio 0.3 > CENSUS_s_bb_0.3_adv_defended2.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_s --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.4 > CENSUS_s_bb_0.4_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_s --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.5 > CENSUS_s_bb_0.5_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_adv --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --test_defended --claim_ratio 0.6 > CENSUS_s_bb_0.6_adv_defended2.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_s --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.7 > CENSUS_s_bb_0.7_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_adv --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --test_defended --claim_ratio 0.8 > CENSUS_s_bb_0.8_adv_defended2.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_s --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.9 > CENSUS_s_bb_0.9_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_s --filter sex --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 1.0 > CENSUS_s_bb_1.0_adv_defended.log


# python -m src.verification --dataset CENSUS --classifier_suffix _defended_r --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.0 > CENSUS_r_bb_0.0_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_r_01 --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover_r_01 --claim_ratio 0.1 > CENSUS_r_bb_0.1_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_r --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.2 > CENSUS_r_bb_0.2_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_r --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.3 > CENSUS_r_bb_0.3_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_r --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.4 > CENSUS_r_bb_0.4_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_r --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.5 > CENSUS_r_bb_0.5_adv_defended.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_adv --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.6 > CENSUS_r_bb_0.6_adv_defended2.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_adv --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.7 > CENSUS_r_bb_0.7_adv_defended2.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_adv --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.8 > CENSUS_r_bb_0.8_adv_defended2.log

# python -m src.verification --dataset CENSUS --classifier_suffix _april_adv --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _april_prover --claim_ratio 0.9 > CENSUS_r_bb_0.9_adv_defended2.log

# python -m src.verification --dataset CENSUS --classifier_suffix _defended_r --filter race --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 1.0 > CENSUS_r_bb_1.0_adv_defended.log