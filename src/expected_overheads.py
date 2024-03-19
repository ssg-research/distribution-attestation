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

import matplotlib.pyplot as plt
import argparse
import numpy as np

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('grid', linestyle="dotted", color='black')



def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ARXIV", help='Options: CENSUS, BONEAGE')
    parser.add_argument('--fixed', type=str, default="FAR", help='Options: FAR,FRR')
    parser.add_argument('--filter', type=str, default="race", help='Options: race, sex')
    args: argparse.Namespace = parser.parse_args()
    return args

args = handle_args()
if args.dataset == 'CENSUS':
    property_vals = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    if args.filter == "race":
        # Original on verification
        # cost_far = [106,173, 359,512,542,551,542,564,389,162,100]
        # cost_frr = [500,660,534,683,894,904,854,799,594,760,505]

        # Robust on verification
        '''NEED TO DO RATIO 0.1'''
        cost_far = [15, 445, 164, 393, 453, 446, 454, 451, 176, 255, 0]
        cost_frr = [5, 271, 69, 253, 391, 438, 372, 304, 96, 210, 5]
        
    else:
        # Original on verification
        # cost_far = [100,434,377,513,547,546,536,545,399,400,100]
        # cost_frr = [500,733,626,688,831,859,861,844,685,685,500]
        

        # Robust on verification
        cost_far = [0, 282, 223, 406, 453, 435, 419, 426, 228, 201, 15]
        cost_frr = [5, 225, 169, 279, 356, 411, 403, 393, 203, 195, 5]

elif args.dataset == 'ARXIV':
    property_vals = ['9', '10', '11', '12', '13', '14', '15', '16', '17']
    # Original on verification
    # cost_far = [437,251,981,613,451,320,374,251,438]
    # cost_frr = [663,768,831,814,786,769,780,798,694]

    # robust on verification
    cost_far = [35, 5, 656, 509, 231, 117, 281, 0]
    cost_frr = [18, 1, 33, 82, 34, 30, 29, 0, 0]

    # robust on test



elif args.dataset == 'BONEAGE':
    property_vals = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
    # Original on verification
    # cost_far =[332,297,396,428,482,296,291]
    # cost_frr = [775,784,707,653,612,488,418]

    # robust on verification
    cost_far = [26, 6, 129, 288, 378, 15, 28]
    cost_frr = [6, 1, 38, 207, 140, 10, 40]





barWidth = 0.25
fig = plt.figure(figsize=(4.5, 3.5))
ax = fig.add_subplot(1, 1, 1)
# ax = fig.add_subplot(1, 1, 1)
# Set position of bar on X axis
br1 = np.arange(len(cost_far))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.bar(br1, cost_far, color ='r', alpha=0.6, width = barWidth,edgecolor ='black', label ='Fixed FAR @ 5%')
# plt.bar(br2, cost_frr, color ='g',  alpha=0.6, width = barWidth,edgecolor ='black', label ='Fixed FRR @ 5%')

# Adding Xticks
plt.xlabel(r"Required Property ($p_{req}$)", fontsize=18)
plt.ylabel("Expected Cost",fontsize=18)
plt.xticks([r for r in range(len(property_vals))],property_vals, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(top=1000)

yticks_loc=ax.get_yticks().tolist()
labels = []
for x in yticks_loc:
    if x == 0:
        labels.append(int(x))
    else:
        labels.append(str(round(int(x)/1000,2)) + r"$\omega_{crpt}$")
ax.set_yticklabels(labels)


# plt.legend()
if args.dataset == "CENSUS":
    plt.savefig(f'{args.dataset}_{args.filter}_expected_cost_rob_verif.pdf', bbox_inches = 'tight',pad_inches = 0, dpi=400)
else:
    plt.savefig(f'{args.dataset}_expected_cost_rob_verif.pdf', bbox_inches = 'tight',pad_inches = 0, dpi=400)
