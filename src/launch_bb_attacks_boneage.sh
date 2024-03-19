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

python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.2 > BONEAGE_bb_0.2_adv_vuln_e8255.log

python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.3 > BONEAGE_bb_0.3_adv_vuln_e8255.log

python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.4 > BONEAGE_bb_0.4_adv_vuln_e8255.log

python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.5 > BONEAGE_bb_0.5_adv_vuln_e8255.log

python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.6 > BONEAGE_bb_0.6_adv_vuln_e8255.log

python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.7 > BONEAGE_bb_0.7_adv_vuln_e8255.log

python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --claim_ratio 0.8 > BONEAGE_bb_0.8_adv_vuln_e8255.log



# python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_defense  --claim_ratio 0.2 > BONEAGE_bb_0.2_adv_gen_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_defense  --claim_ratio 0.3 > BONEAGE_bb_0.3_adv_gen_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_defense  --claim_ratio 0.4 > BONEAGE_bb_0.4_adv_gen_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_defense  --claim_ratio 0.5 > BONEAGE_bb_0.5_adv_gen_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_defense  --claim_ratio 0.6 > BONEAGE_bb_0.6_adv_gen_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_defense  --claim_ratio 0.7 > BONEAGE_bb_0.7_adv_gen_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_defense  --claim_ratio 0.8 > BONEAGE_bb_0.8_adv_gen_e8255.log



# python -m src.verification --dataset BONEAGE --classifier_suffix _defended_e8255 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.2 > BONEAGE_bb_0.2_adv_defended_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _defended_e8255 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.3 > BONEAGE_bb_0.3_adv_defended_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _defended_e8255 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.4 > BONEAGE_bb_0.4_adv_defended_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _defended_e8255 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.5 > BONEAGE_bb_0.5_adv_defended_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _defended_e8255 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.6 > BONEAGE_bb_0.6_adv_defended_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _defended_e8255 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.7 > BONEAGE_bb_0.7_adv_defended_e8255.log

# python -m src.verification --dataset BONEAGE --classifier_suffix _defended_e8255 --do_attack --bb_adv_attack --classifier_suffix_prover _prover --test_defended --claim_ratio 0.8 > BONEAGE_bb_0.8_adv_defended_e8255.log
