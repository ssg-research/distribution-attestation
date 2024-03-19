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

import numpy as np
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
import argparse
import pickle

def normalize(args, value, scale_factor = 1000):
    if args.normalization_maximum > 0:
        return value / args.normalization_maximum * scale_factor
    return value
    

def make_hybrid_plots(args):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    plt.rc('grid', linestyle="dotted", color='black')

    M = 1000 # total number of objects (provers)
    n = args.num_false_accepts

    N = np.arange(0, args.number_of_accepts+1)
    print(f'Total accepts: {args.number_of_accepts}') 
   

    new_far = []
    total_costs = []
    total_cost_cryto = []
    print(f'{args.specific_metric} with no crypto: {args.far_inference}')
    print(f'Cost with no crypto: {M*args.online_cost_inf}')
    for i, num_select in enumerate(N):
        rv = hypergeom(args.number_of_accepts, n, num_select)
        x = np.arange(0, n+1) # All the possible numbers of false accepts we can catch
        pmf_fa = rv.pmf(x)
        
        x_prime = np.argmax(pmf_fa)   # check
        fa_new = n - x_prime
        new_far.append(fa_new/args.num_actual_negative)

        if args.plot_pmfs:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, pmf_fa, 'bo')
            ax.vlines(x, 0, pmf_fa, lw=2)
            ax.set_xlabel('# of false accepts in our group of chosen provers')
            ax.set_ylabel('hypergeom PMF')
            plt.show()
            plt.savefig(f'{args.dataset}_{args.filter}_{args.claim_property}_hypergeom_pmf_{i}_samples_taken.png')
        
        if num_select == args.number_of_accepts:
            print(f'{args.specific_metric} with full crypto: {new_far[-1]}')
            print(f'Cost with full crypto: {M * args.online_cost_inf}+{normalize(args, num_select, M)}C')
        
        if num_select == n:
            print(f'{args.specific_metric} with sampling {n} (1-prec): {new_far[-1]}')
            print(f'Cost with with sampling {n} (1-prec): {M * args.online_cost_inf}+{normalize(args, num_select, M)}C')
        
        # break

    total_cost_inference = [round(M * args.online_cost_inf,2)]*len(new_far)

    # Change the y_ticks to have the character instead
    # Need to normalize cost
    total_costs_labels = [i for i, inference_cost in enumerate(total_cost_inference)]
    total_costs = [normalize(args, val, M) for val in range(len(total_costs_labels))]
    # total_costs = [crypto_cost + total_cost_inference[i] for i, crypto_cost in enumerate(total_cost_cryto)]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax = fig.add_subplot(111)
    N_normalized = []
    for num_select in N:
        N_normalized.append(normalize(args, num_select, M))
    ax1.plot(N_normalized, new_far, 'go', label=f'Effective {args.specific_metric}')
    ax1.set_xlabel('Number of Spot-Checks')
    ax1.set_ylabel('Effective FAR')
    # plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax2.plot(N_normalized, total_costs, 'bo', label='Cost')
    # ax.set_xlabel('Proportion of Spot-Checks by Verifier')
    ax2.set_ylabel('Crypto Cost Coefficient (# of provers)')
    # ax2.set_yticks(xrange(len(total_costs_labels)))
    # ax2.set_yticklabels(total_costs_labels)
    # ax2.set_yticks(total_costs, total_costs)
    ax2.legend(loc='lower center')
    ax1.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{args.dataset}_{args.filter}_{args.claim_property}_total_cost_and_new_{args.specific_metric}.png')

    # print(f'new_far: {new_far}')
    # print(f'total costs: {total_costs}')
    claim_property = str(args.claim_property)
    suffix = 'test' if args.parse_test else 'verify'
    if args.dataset == 'ARXIV' and args.claim_property == 9:
        claim_property = '09'
    with open(f'{args.dataset}_{args.filter}_{claim_property}_{args.specific_metric}_newfar_{suffix}.pkl', 'wb') as file:
        pickle.dump(new_far, file)

    with open(f'{args.dataset}_{args.filter}_{claim_property}_{args.specific_metric}_total_costs_{suffix}.pkl', 'wb') as file:
        pickle.dump(total_costs, file)


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--online_cost_crypto', type=float, default=38000.)
    parser.add_argument('--online_cost_inf', type=float, default=30.) # WE NEED TO MEASURE THIS
    parser.add_argument('--dataset', default='BONEAGE', type=str)
    parser.add_argument('--filter', default='', type=str)
    parser.add_argument('--claim_property', default=0.5, type=float)
    parser.add_argument('--far_inference', default=0.52, type=float)
    parser.add_argument('--tar_inference', default=0.24, type=float)
    parser.add_argument('--specific_metric', default='FAR', type=str, help='We may want to plot either the (effective) FAR or FRR')
    parser.add_argument('--plot_pmfs', action='store_true', default=False)
    parser.add_argument('--number_of_accepts', default=1200, type=int, help='The total number of acceptances (TA+FA) at the threshold (eg. at TAR 5%)')
    parser.add_argument('--num_actual_negative', default = 600, type=int, help = 'The actual number of rejects (TR + FA) at the threshold (eg. at TAR 5%)')
    parser.add_argument('--num_false_accepts', type=int, help='Number of false accepts at the threshold (eg. at TAR 5%)')
    parser.add_arugment('--normalization_maximum', type=int, default = 0, help='When normalization is required, the normalization maximum is set. It is assumed that the normalization minimum is 0')
    args: argparse.Namespace = parser.parse_args()
    return args
    

if __name__=='__main__':
    args = handle_args()
    make_hybrid_plots(args)