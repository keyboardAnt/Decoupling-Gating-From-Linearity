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
parser.add_argument("--target_loss", type=float, nargs="+", default=[0.3])

parser.add_argument("--lr", type=float, nargs="+", default=[0.001])
parser.add_argument("--iters", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--min_k", type=int, default=1)
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

criterion = nn.MSELoss()
combinations = list(product(args.T, args.m, args.d, args.target_loss, args.d_,
    args.network_type, args.lr))

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# binary_search_size = int(np.log2(args.max_k - args.min_k + 1)) + 2

max_k = args.max_k
max_d = args.d[0]

heatmap = np.zeros((max_k, max_d))

for T, m, d, target_loss, d_, network_type, lr in tqdm(combinations):
    # min_k, max_k = args.min_k, args.max_k
    # tested_k = set()
#    while (max_k - min_k) > 0:
    for k in range(1, heatmap.shape[0]):
        for d in range(1, heatmap.shape[1]):
            print(f'k={k}, d={d}')
    # for i in tqdm(range(binary_search_size)):
    #     k = (max_k + min_k) // 2
    #     if k in tested_k:
    #         k = k + 1
    #         if k in tested_k:
    #             break
    #     tested_k.add(k)

            train_loss = 0.

            # if args.seed is not None:
            #     torch.manual_seed(args.seed)
            #     np.random.seed(args.seed)

            for j in tqdm(range(args.repeat)):
                X = torch.randn(m, d)
                Y = torch.randn(m, d_)
                # Y = torch.bernoulli(.5, out=(m, d_))

                network = networks.NETWORKS[network_type](d, d_, k, T)
                network.optimize(X, Y, criterion, args.batch_size, args.iters, lr, tqdm)

                Y_hat = network(X)
                train_loss += criterion(Y_hat, Y)

            train_loss /= args.repeat

            # with open(args.output_file, "a") as f:
            #     print(title.format(network_type, T, m, d, k, d_, lr), file=f)
            #     print("=" * len(title), file=f)
            #     print("loss={:.2e}".format(train_loss), file=f)
            #     print("\n", file=f)

            # if train_loss > target_loss:
            #     min_k = k
            # if train_loss < target_loss:
            #     max_k = k
            heatmap[k - 1, d - 1] = train_loss

heatmap_filename = 'heatmap_network_type={}, T={}, m={}, d={}, k={}, d_={}, lr={:.2e}.npy'\
    .format(network_type, T, m, d, k, d_, lr)
RESULTS_DIR = 'results'
heatmap_filepath = os.path.join(RESULTS_DIR, heatmap_filename)
with open(heatmap_filepath, 'wb') as f:
    np.save(f, heatmap)
