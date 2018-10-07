from argparse import ArgumentParser
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

import networks


parser = ArgumentParser()
parser.add_argument("output_file")

parser.add_argument("network_type", nargs="+", choices=["relu", "galu", "galu_opt"])
parser.add_argument("-m", type=int, nargs="+", default=[1024])
parser.add_argument("-d", type=int, nargs="+", default=[16])
parser.add_argument("-k", type=int, nargs="+", default=[16])

parser.add_argument("--hill_climb_iters", type=int, nargs="+", default=[128])

parser.add_argument("--sgd_lr", type=float, nargs="+", default=[0.001])
parser.add_argument("--sgd_iters", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--seed", type=int)
parser.add_argument("--tqdm", type=int, default=1)

args = parser.parse_args()

if args.tqdm:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x
else:
    tqdm = lambda x: x

with open(args.output_file, "x") as f:
    print(args, file=f)
    print("sgd_iters={}".format(args.sgd_iters), file=f)
    print("batch_size={}".format(args.batch_size), file=f)
    print("repeat={}".format(args.repeat), file=f)
    print("seed={}".format(args.seed), file=f)
    print("+" * 80, file=f)
    print("\n", file=f)
title = "network_type={}, m={}, d={}, k={}, lr={:.2e}, hill_climb_iters={}"

criterion = nn.MSELoss()

network_params = []
for network_type in args.network_type:
    if network_type == "galu_opt":
        network_params.extend(product([network_type], [-1], args.hill_climb_iters))
    else:
        network_params.extend(product([network_type], args.sgd_lr, [-1]))

combinations_ = list(product(args.m, args.d, args.k, network_params))
combinations = []
for m, d, k, (network_type, lr, hci) in combinations_:
    if d * k <= m:
        combinations.append((m, d, k, (network_type, lr, hci)))

for m, d, k, (network_type, lr, hci) in tqdm(combinations):
    train_loss = 0.

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    for i in tqdm(range(args.repeat)):
        X = torch.randn(m, d)
        Y = torch.randn(m, 1)

        if network_type == "relu":
            network = networks.ReLUNetwork(d, 1, k, 1, bias=False)
            network.optimize(X, Y, criterion, args.batch_size, args.sgd_iters,
                lr, tqdm)

        elif network_type == "galu":
            network = networks.GaLUNetwork(d, 1, k, 1, bias=False)
            network.optimize(X, Y, criterion, args.batch_size, args.sgd_iters,
                lr, tqdm)

        elif network_type == "galu_opt":
            network = networks.GaLUNetwork_Shallow_R1(d, k)
            network.hill_climb(X, Y, hci, tqdm)

        Y_hat = network(X)
        train_loss += criterion(Y_hat, Y)

    with open(args.output_file, "a") as f:
        print(title.format(network_type, m, d, k, lr, hci), file=f)
        print("=" * len(title), file=f)
        print("loss={:.2e}".format(train_loss / args.repeat), file=f)
        print("\n", file=f)
