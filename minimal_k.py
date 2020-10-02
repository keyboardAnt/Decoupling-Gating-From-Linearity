from argparse import ArgumentParser
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import os

import networks

parser = ArgumentParser()
parser.add_argument("output_file")
parser.add_argument("network_type", nargs="+", choices=["relu", "galu",
    "galu0"])
parser.add_argument("-m", type=int, nargs="+", default=[1024])
parser.add_argument("-d", type=int, nargs="+", default=[100])
parser.add_argument("-d_", type=int, nargs="+", default=[10])
parser.add_argument("-T", type=int, nargs="+", default=[1])
parser.add_argument("--target_loss", type=float, nargs="+", default=[0.3]) # Not supported anymore

parser.add_argument("--lr", type=float, nargs="+", default=[0.001])
parser.add_argument("--iters", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--min_k", type=int, default=1) # Not supported anymore
parser.add_argument("--max_k", type=int, default=512)

parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--tqdm", type=int, default=1)

args = parser.parse_args()
print(args)

if args.tqdm:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x
else:
    tqdm = lambda x: x

with open(args.output_file, "w") as f:
    print(args, file=f)
    print("iters={}".format(args.iters), file=f)
    print("batch_size={}".format(args.batch_size), file=f)
    print("repeat={}".format(args.repeat), file=f)
    print("seed={}".format(args.seed), file=f)
    print("+" * 80, file=f)
    print("\n", file=f)
title = "network_type={}, T={}, m={}, d={}, k={}, d_={}, lr={:.2e}"


def _set_seeds(seed: int) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


_set_seeds(args.seed)

criterion = nn.MSELoss()

max_k = args.max_k
max_d = args.d[0]
heatmap = np.zeros((max_k, max_d))


combinations = list(product(args.T, args.m, args.d, args.target_loss, args.d_, args.network_type, args.lr))
for T, m, d, target_loss, d_, network_type, lr in tqdm(combinations):
    # Generate the data labels
    # Y = torch.empty(m, d_).fill_(.5)
    # Y = torch.bernoulli(Y)
    Y = torch.randn(m, d_)
    for d in range(1, heatmap.shape[1] + 1):
        # Generate the data inputs
        X = torch.randn(m, d)
        for k in range(1, heatmap.shape[0] + 1):
            print(f'k={k}, d={d}')
            train_loss = 0.
            network = networks.NETWORKS[network_type](d, d_, k, T)
            network.optimize(X, Y, criterion, args.batch_size, args.iters, lr, tqdm)
            Y_hat = network(X)
            train_loss += criterion(Y_hat, Y)
            train_loss /= args.repeat
            heatmap[k - 1, d - 1] = train_loss

heatmap_filename = 'heatmap_network_type={}, T={}, m={}, d={}, k={}, d_={}, lr={:.2e}.npy'\
    .format(network_type, T, m, d, k, d_, lr)
RESULTS_DIR = 'results'
heatmap_filepath = os.path.join(RESULTS_DIR, heatmap_filename)
with open(heatmap_filepath, 'wb') as f:
    np.save(f, heatmap)
