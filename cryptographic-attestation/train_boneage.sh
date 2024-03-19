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


python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi1.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi2.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi3.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi4.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi5.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi6.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi7.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi8.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi9.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi10.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi11.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi12.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi13.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi14.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi15.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi16.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi17.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi18.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi19.log
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten > bone_run_multi20.log


# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run1.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run2.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run3.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run4.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run5.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run6.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run7.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run8.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run9.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run10.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run11.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run12.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run13.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run14.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run15.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run16.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run17.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run18.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run19.log
# python bone_model_training.py --gpu=0 --crypten --multiprocess > bone_run20.log

# python bone_model_training.py --gpu=0 --crypten  > bone_run1_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run2_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run3_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run4_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run5_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run6_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run7_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run8_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run9_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run10_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run11_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run12_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run13_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run14_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run15_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run16_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run17_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run18_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run19_crypten_acc.log
# python bone_model_training.py --gpu=0 --crypten  > bone_run20_crypten_acc.log

# python bone_model_training.py --gpu=0   > bone_run1_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run2_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run3_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run4_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run5_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run6_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run7_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run8_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run9_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run10_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run11_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run12_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run13_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run14_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run15_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run16_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run17_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run18_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run19_noncrypten_acc.log
# python bone_model_training.py --gpu=0   > bone_run20_noncrypten_acc.log