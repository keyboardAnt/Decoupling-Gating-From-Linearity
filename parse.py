from argparse import Namespace
from collections import defaultdict
import re
import numpy as np

HEADER_LINE = re.compile(r"^((\w+)=([\w\.\-\+]+)(?:, )?){2,}$")
HEADER_LINE_P = re.compile(r"(\w+)=([\w\.\-\+]+)")
FLOAT = re.compile(r"([\w\.]+e[\+\-]\w+)")

TYPES = {
    "m": int,
    "d": int,
    "d_": int,
    "k": int,
    "o": int,
    "T": int,
    "hill_climb_iters": int,
    "lr": float,
    "a": float,
    "n": int,
    "sigma_x": float,
    "sigma_y": float,
}

def fix_dict(d):
    for k, t in TYPES.items():
        if k in d:
            d[k] = t(d[k])

class Results:
    def __init__(self, filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.results = []
        for filename in filenames:
            with open(filename) as f:
                self.namespace = eval(f.readline())
                for line in f:
                    if set(line[:-1]) == {"+"}:
                        break
                for line in f:
                    if len(set(line)) > 2:
                        if HEADER_LINE.match(line):
                            header = dict(HEADER_LINE_P.findall(line))
                            self.results.append((header, []))
                        else:
                            self.results[-1][1].append(line)

    def parse_memorization(self):
        new_results = []
        for header, data in self.results:
            fix_dict(header)
            loss = float(HEADER_LINE_P.findall(data[0])[0][1])
            data = {"loss": loss}
            new_results.append((header, data))
        self.results = new_results

    def parse_datasets(self):
        new_results = []
        for header, data in self.results:
            fix_dict(header)
            (_, train_ce), (_, train_01) = HEADER_LINE_P.findall(data[0])
            (_, test_ce), (_, test_01) = HEADER_LINE_P.findall(data[1])
            data = {
                "train_ce": float(train_ce),
                "train_01": float(train_01),
                "test_ce": float(test_ce),
                "test_01": float(test_01)
            }
            new_results.append((header, data))
        self.results = new_results

    def parse_generalization(self):
        new_results = []
        for header, data in self.results:
            fix_dict(header)
            (_, train_ce), (_, train_01) = HEADER_LINE_P.findall(data[0])
            (_, test_ce), (_, test_01) = HEADER_LINE_P.findall(data[1])
            (_, l1), (_, l2) = HEADER_LINE_P.findall(data[2])
            data = {
                "train_ce": float(train_ce),
                "train_01": float(train_01),
                "test_ce": float(test_ce),
                "test_01": float(test_01),
                "l1": float(l1),
                "l2": float(l2)
            }
            new_results.append((header, data))
        self.results = new_results

    def parse_realizable(self):
        new_results = []
        for header, data in self.results:
            fix_dict(header)
            train = FLOAT.findall(data[0])[0]
            test = FLOAT.findall(data[1])[0]
            data = {
                "train": float(train),
                "test" : float(test)
            }
            new_results.append((header, data))
        self.results = new_results

    def best_groupby(self, by, criterion):
        keys = list(self.results[0][0].keys())
        for b in by:
            keys.remove(b)
        D = defaultdict(dict)
        for header, data in self.results:
            key = tuple(header[k] for k in keys)
            D[key][tuple(header[b] for b in by)] = data
        new_results = []
        for key in D:
            best = float("inf")
            best_val = None
            for k in D[key]:
                if D[key][k][criterion] < best:
                    best = D[key][k][criterion]
                    best_val = D[key][k]
            new_results.append((dict(zip(keys, key)), best_val))
        self.results = new_results

    def best_lr(self, criterion):
        self.best_groupby(["lr"], criterion)

    def best_lr_a(self, criterion):
        self.best_groupby(["lr", "a"], criterion)


    def select(self, select):
        for header, data in self.results:
            if all(header[k] == select[k] for k in select):
                yield header, data

    def plot_results(self, select, x, y):
        X = []
        Y = []
        if isinstance(x, str):
            f = lambda header: header[x]
        else:
            f = x
        for header, data in self.select(select):
            X.append(f(header))
            Y.append(data[y])
        X = np.array(X)
        Y = np.array(Y)
        return X[X.argsort()], Y[X.argsort()]

