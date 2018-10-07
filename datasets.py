from argparse import ArgumentParser
from itertools import product

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import networks
import data_lib


parser = ArgumentParser()
parser.add_argument("output_file")

parser.add_argument("network_type", nargs="+",
    choices=list(networks.NETWORKS.keys()))
parser.add_argument("-d", type=int, default=64)
parser.add_argument("-k", type=int, nargs="+", default=[64])
parser.add_argument("-T", type=int, default=2)
parser.add_argument("--dataset", nargs="+", default=["mnist"])

parser.add_argument("--lr", type=float, nargs="+", default=[0.001])
parser.add_argument("--iters", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("-a", type=float, nargs="+", default=1.)
parser.add_argument("--only_readout", type=int, default=0)

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
    print("d={}".format(args.d), file=f)
    print("T={}".format(args.T), file=f)
    print("iters={}".format(args.iters), file=f)
    print("batch_size={}".format(args.batch_size), file=f)
    print("repeat={}".format(args.repeat), file=f)
    print("seed={}".format(args.seed), file=f)
    print("only_readout={}".format(args.only_readout), file=f)
    print("+" * 80, file=f)
    print("\n", file=f)
title = "dataset={}, network_type={}, k={}, a={:.2e}, lr={:.2e}"

criterion = CrossEntropyLoss()

network_a = []
for network_type in args.network_type:
    if network_type in ["galus", "galu0s"]:
        network_a.extend(product([network_type], args.a))
    else:
        network_a.append((network_type, 1.0))

datasets = {}
for dataset in args.dataset:
    datasets[dataset] = data_lib.pca_dataset(dataset, args.d)

combinations = list(product(args.dataset, args.k, network_a, args.lr))

for dataset, k, (network_type, a), lr in tqdm(combinations):
    train_ce = 0.
    train_01 = 0.
    test_ce = 0.
    test_01 = 0.
    (X, Y), (X_test, Y_test) = datasets[dataset]

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    for i in tqdm(range(args.repeat)):
        network = networks.NETWORKS[network_type](args.d, 10, k, args.T,
            alpha=a)
        network.optimize(X, Y, criterion, args.batch_size, args.iters, lr, tqdm)

        Y_hat = network(X)
        train_ce += criterion(Y_hat, Y)
        train_01 += (Y_hat.argmax(1) != Y).to(torch.float32).mean()

        Y_hat_test = network(X_test)
        test_ce += criterion(Y_hat_test, Y_test)
        test_01 += (Y_hat_test.argmax(1) != Y_test).to(torch.float32).mean()

    with open(args.output_file, "a") as f:
        print(title.format(dataset, network_type, k, a, lr), file=f)
        print("=" * len(title), file=f)
        print("Train: ce={:.2e}, 01={:.2e}".format(train_ce / args.repeat,
            train_01 / args.repeat), file=f)
        print("Test : ce={:.2e}, 01={:.2e}".format(test_ce / args.repeat,
            test_01 / args.repeat), file=f)
        print("\n", file=f)
