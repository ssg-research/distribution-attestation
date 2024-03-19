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

import os
import matplotlib.pyplot as plt
import argparse
import pickle

def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ARXIV", help='Options: CENSUS, BONEAGE')
    parser.add_argument('--filter', type=str, default="r", help='Options: r, s')
    parser.add_argument('--test', action='store_true', default = False, help='Plot the FAR-cost plot derived fro mthe verifier test dataset')
    args: argparse.Namespace = parser.parse_args()
    return args

args = handle_args()

dirs = os.listdir('.')
dirs.sort()
new_far_files = [file for file in dirs if file.startswith(args.dataset) and file.endswith(f'newfar_{args.test}.pkl')]
total_costs_files = [file for file in dirs if file.startswith(args.dataset) and file.endswith(f'total_costs_{args.test}.pkl')]

if args.dataset == 'CENSUS':
    new_far_files = [file for file in new_far_files if file.startswith(f'{args.dataset}__{args.filter}')]
    total_costs_files = [file for file in total_costs_files if file.startswith(f'{args.dataset}__{args.filter}')]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF5733', '#008080', '#800080', '#FFD700']

fig = plt.figure(figsize=(4.5, 3.5))
ax = fig.add_subplot(1, 1, 1)
all_fars = []
all_costs = []
print(new_far_files)
print()
for i, (farfile, costfile) in enumerate(zip(new_far_files, total_costs_files)):
    print(f'{farfile}, {costfile}')
    with open(farfile, 'rb') as f:
        fars = pickle.load(f)
    with open(costfile, 'rb') as c:
        costs = pickle.load(c)
    all_fars.append(fars)
    all_costs.append(costs)
    if args.dataset == 'CENSUS':
        label = farfile.split(f'{args.dataset}__{args.filter}_')[-1].split('_')[0]
    else:
        label = farfile.split(f'{args.dataset}__')[-1].split('_')[0]
        if label == '09' and args.dataset == 'ARXIV':
            label = '9'
    plt.plot(fars, costs, color=colors[i], label=label, linewidth = 3)

plt.ylim(0, 1000)
plt.xlim(0, 0.3)
plt.xlabel(r"FAR", fontsize=18)
plt.ylabel("Expected Cost",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

yticks_loc=ax.get_yticks().tolist()
labels = []
for x in yticks_loc:
    if x == 0:
        labels.append(int(x))
    else:
        labels.append(str(round(int(x)/1000,2)) + r"$\omega_{crpt}$")
ax.set_yticklabels(labels)

plt.grid(True)
legend_fontsize = 13
if args.dataset == 'BONEAGE':
    legend_fontsize = 14
elif args.dataset == 'CENSUS':
    legend_fontsize = 11

# if args.dataset == 'CENSUS' and args.filter == 's' and not args.test:
#     legend = plt.legend(frameon=True, title=r'$P_{req}$', loc='upper left') # bbox_to_anchor=(0.9, 0.1)
# else:
legend = plt.legend(frameon=True, title=r'$P_{req}$', fontsize=legend_fontsize, loc='upper right') # bbox_to_anchor=(0.9, 0.1)
legend.get_frame().set_alpha(0.7)
suffix = 'verify'
if args.test:
    suffix = 'test'

if args.dataset == 'CENSUS':
    plt.savefig(f'{args.dataset}_{args.filter}_cost_far_{suffix}.pdf', bbox_inches = 'tight',pad_inches = 0, dpi=400)
else:
    plt.savefig(f'{args.dataset}_cost_far_{suffix}.pdf', bbox_inches = 'tight',pad_inches = 0, dpi=400)