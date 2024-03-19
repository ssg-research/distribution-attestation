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

echo 9 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 9 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log

echo 10 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 10 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log

echo 11 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 11 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log

echo 12 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 12 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log

echo 13 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 13 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log

echo 14 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 14 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log

echo 15 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 15 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log

echo 16 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 16 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log

echo 17 >> ARXIV_finetuned_checks.log
python -m src.parse_logs --dataset ARXIV --claim_property 17 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log


echo 0.2 >> BONEAGE_finetuned_checks.log
python -m src.parse_logs --dataset BONEAGE --claim_property 0.2 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> BONEAGE_finetuned_checks.log

echo 0.3 >> BONEAGE_finetuned_checks.log
python -m src.parse_logs --dataset BONEAGE --claim_property 0.3 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> BONEAGE_finetuned_checks.log

echo 0.4 >> BONEAGE_finetuned_checks.log
python -m src.parse_logs --dataset BONEAGE --claim_property 0.4 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> BONEAGE_finetuned_checks.log

echo 0.5 >> BONEAGE_finetuned_checks.log
python -m src.parse_logs --dataset BONEAGE --claim_property 0.5 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> BONEAGE_finetuned_checks.log

echo 0.6 >> BONEAGE_finetuned_checks.log
python -m src.parse_logs --dataset BONEAGE --claim_property 0.6 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> BONEAGE_finetuned_checks.log

echo 0.7 >> BONEAGE_finetuned_checks.log
python -m src.parse_logs --dataset BONEAGE --claim_property 0.7 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> BONEAGE_finetuned_checks.log

echo 0.8 >> BONEAGE_finetuned_checks.log
python -m src.parse_logs --dataset BONEAGE --claim_property 0.8 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> BONEAGE_finetuned_checks.log


echo 0.0 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.0 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.1 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.1 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.2 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.2 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.3 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.3 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.4 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.4 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.5 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.5 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.6 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.6 --logfile_suffix _adv_defended2 --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.7 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.7 --logfile_suffix _adv_defended2 --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.8 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.8 --logfile_suffix _adv_defended2 --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 0.9 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 0.9 --logfile_suffix _adv_defended2 --finetuned_performance_check >> CENSUS_r_finetuned_checks.log

echo 1.0 >> CENSUS_r_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter race --claim_property 1.0 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_r_finetuned_checks.log



echo 0.0 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.0 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.1 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.1 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.2 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.2 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.3 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.3 --logfile_suffix _adv_defended2 --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.4 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.4 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.5 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.5 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.6 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.6 --logfile_suffix _adv_defended2 --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.7 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.7 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.8 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.8 --logfile_suffix _adv_defended2 --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 0.9 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 0.9 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_s_finetuned_checks.log

echo 1.0 >> CENSUS_s_finetuned_checks.log
python -m src.parse_logs --dataset CENSUS --filter sex --claim_property 1.0 --logfile_suffix _adv_defended --finetuned_performance_check >> CENSUS_s_finetuned_checks.log



