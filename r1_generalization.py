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
parser.add_argument("--problem_type", nargs="+", choices=["regression", "classification"])
parser.add_argument("-m", type=int, nargs="+", default=[1000])
parser.add_argument("-d", type=int, nargs="+", default=[30])
parser.add_argument("-k", type=int, nargs="+", default=[30])
parser.add_argument("-n", type=int, nargs="+", default=[30])

parser.add_argument("--sigma_y", type=float, nargs="+", default=[0.1])
parser.add_argument("--sigma_x", type=float, nargs="+", default=[0.1])

parser.add_argument("--hill_climb_iters", type=int, nargs="+", default=[0])
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
title = "network_type={}, problem_type={}, sigma_x={}, sigma_y={}, m={}, d={}, k={}, n={}, lr={:.2e}"

network_params = []
for network_type, problem_type in product(args.network_type, args.problem_type):
    if network_type in {"relu", "galu"}:
        network_params.extend(product([network_type], [problem_type], args.sgd_lr))
    elif problem_type == "regression":
        network_params.append(("galu_opt", "regression", -1))


combinations = list(product(args.m, args.d, args.k, args.n, args.sigma_x, args.sigma_y, network_params))


def generate_classication_data(m, d, n, sigma_x, sigma_y):
    c = np.random.randn(n, d)
    b = np.sign(np.random.randn(n))
    i = np.random.choice(np.arange(n), 10 * m, replace=True)

    label_noise = -1. + 2. * (np.random.random((10 * m, 1)) > 0.5 * sigma_y)

    X = torch.Tensor(c[i, :   ] + sigma_x * np.random.randn(10 * m, d))
    Y = torch.Tensor(b[i, None] * label_noise)

    return (X[:m], Y[:m]), (X[m:], Y[m:])

def generate_regression_data(m, d, n, sigma_x, sigma_y):
    c = np.random.randn(n, d)
    b = np.random.randn(n)
    i = np.random.choice(np.arange(n), 10 * m, replace=True)

    label_noise = sigma_y * np.random.randn(10 * m, 1)

    X = torch.Tensor(c[i, :   ] + sigma_x * np.random.randn(10 * m, d))
    Y = torch.Tensor(b[i, None] + label_noise)

    return (X[:m], Y[:m]), (X[m:], Y[m:])


for m, d, k, n, sigma_x, sigma_y, (network_type, problem_type, lr) in tqdm(combinations):
    train_loss = 0.
    train_01 = 0.
    test_loss = 0.
    test_01 = 0.
    l1_norm = 0.
    l2_norm = 0.

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    for i in tqdm(range(args.repeat)):
        if problem_type == "classification":
            (X_train, Y_train), (X_test, Y_test) = generate_classication_data(m, d, n, sigma_x, sigma_y)
            criterion = lambda x, y, z=torch.tensor([0.0]): torch.max(1.0 - x * y, z).mean()
        elif problem_type == "regression":
            (X_train, Y_train), (X_test, Y_test) = generate_regression_data(m, d, n, sigma_x, sigma_y)
            criterion = nn.MSELoss()

        if network_type == "galu_opt":
            network = networks.GaLUNetwork_Shallow_R1(d, k)
            network.optimize(X_train, Y_train)
        else:
            if network_type == "relu":
                network = networks.ReLUNetwork(d, 1, k, 1, bias=False)

            elif network_type == "galu":
                network = networks.GaLUNetwork(d, 1, k, 1, bias=False)

            network.optimize(X_train, Y_train, criterion, args.batch_size,
                args.sgd_iters, lr, tqdm)

        Y_hat = network(X_train)
        train_loss += criterion(Y_hat, Y_train)
        train_01 += (Y_hat.sign() != Y_train).to(torch.float32).mean()

        Y_hat_test = network(X_test)
        test_loss += criterion(Y_hat_test, Y_test)
        test_01 += (Y_hat_test.sign() != Y_test).to(torch.float32).mean()

        l1_norm += network.norm(1)
        l2_norm += network.norm(2)

    with open(args.output_file, "a") as f:
        print(title.format(network_type, problem_type, sigma_x, sigma_y, m, d, k, n, lr), file=f)
        print("=" * len(title), file=f)
        print("Train: ce={:.2e}, 01={:.2e}".format(train_loss / args.repeat,
            train_01 / args.repeat), file=f)
        print("Test : ce={:.2e}, 01={:.2e}".format(test_loss / args.repeat,
            test_01 / args.repeat), file=f)
        print("Norm : l1={:.2e}, l2={:.2e}".format(l1_norm / args.repeat,
            l2_norm / args.repeat), file=f)
        print("\n", file=f)
