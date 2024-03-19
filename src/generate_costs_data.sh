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

python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 9 > ARXIV9_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 10 > ARXIV10_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 11 > ARXIV11_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 12 > ARXIV12_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 13 > ARXIV13_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 14 > ARXIV14_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 15 > ARXIV15_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 16 > ARXIV16_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis  --claim_property 17 > ARXIV17_hybrid_analysis_plotting_verify.log


python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis  --claim_property 0.2 > BONEAGE02_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis  --claim_property 0.3 > BONEAGE03_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis  --claim_property 0.4 > BONEAGE04_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis  --claim_property 0.5 > BONEAGE05_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis  --claim_property 0.6 > BONEAGE06_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis  --claim_property 0.7 > BONEAGE07_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis  --claim_property 0.8 > BONEAGE08_hybrid_analysis_plotting_verify.log


python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.0 > CENSUS00_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.1 > CENSUS01_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.2 > CENSUS02_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.3 > CENSUS03_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.4 > CENSUS04_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.5 > CENSUS05_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.6 > CENSUS06_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.7 > CENSUS07_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.8 > CENSUS08_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 0.9 > CENSUS09_r_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis  --claim_property 1.0 > CENSUS10_r_hybrid_analysis_plotting_verify.log

python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.0 > CENSUS00_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.1 > CENSUS01_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.2 > CENSUS02_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.3 > CENSUS03_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.4 > CENSUS04_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.5 > CENSUS05_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.6 > CENSUS06_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.7 > CENSUS07_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.8 > CENSUS08_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 0.9 > CENSUS09_s_hybrid_analysis_plotting_verify.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis  --claim_property 1.0 > CENSUS10_s_hybrid_analysis_plotting_verify.log




python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 9 > ARXIV9_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 10 > ARXIV10_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 11 > ARXIV11_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 12 > ARXIV12_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 13 > ARXIV13_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 14 > ARXIV14_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 15 > ARXIV15_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 16 > ARXIV16_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset ARXIV --hybrid_analysis --parse_test --claim_property 17 > ARXIV17_hybrid_analysis_plotting_test.log


python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis --parse_test --claim_property 0.2 > BONEAGE02_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis --parse_test --claim_property 0.3 > BONEAGE03_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis --parse_test --claim_property 0.4 > BONEAGE04_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis --parse_test --claim_property 0.5 > BONEAGE05_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis --parse_test --claim_property 0.6 > BONEAGE06_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis --parse_test --claim_property 0.7 > BONEAGE07_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended_e8255" --logfile_subdir "baseline_defended_aug8" --dataset BONEAGE --hybrid_analysis --parse_test --claim_property 0.8 > BONEAGE08_hybrid_analysis_plotting_test.log


python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.0 > CENSUS00_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.1 > CENSUS01_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.2 > CENSUS02_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.3 > CENSUS03_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.4 > CENSUS04_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.5 > CENSUS05_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.6 > CENSUS06_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.7 > CENSUS07_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.8 > CENSUS08_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 0.9 > CENSUS09_r_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter race --hybrid_analysis --parse_test --claim_property 1.0 > CENSUS10_r_hybrid_analysis_plotting_test.log

python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.0 > CENSUS00_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.1 > CENSUS01_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.2 > CENSUS02_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.3 > CENSUS03_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.4 > CENSUS04_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.5 > CENSUS05_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.6 > CENSUS06_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.7 > CENSUS07_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.8 > CENSUS08_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.9 > CENSUS09_s_hybrid_analysis_plotting_test.log
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 1.0 > CENSUS10_s_hybrid_analysis_plotting_test.log
