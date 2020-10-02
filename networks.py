from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# from torch.optim import Adam
from torch.optim import SGD


NON_LINEARITIES = {
    "none": lambda x: x,
    "gate": lambda x: (torch.sign(x) + 1.) / 2.,
    "relu": F.relu,
    "sign": torch.sign,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid
}


class Batches:
    """
    Manages mini-batches: returns the indeces of examples according to random
    permutations.
    """

    def __init__(self, sample_size, batch_size, iters):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.p = np.random.permutation(self.sample_size)
        self.ind = 0
        self.reminder = iters

    def __iter__(self):
        return self

    def __len__(self):
        return self.reminder

    def __next__(self):
        if self.reminder:
            end = self.ind + self.batch_size
            if (end < self.sample_size):
                ret = self.p[self.ind : end]
                self.ind = end
            else:
                ret = self.p[self.ind:]
                self.p = np.random.permutation(self.sample_size)
                self.ind = end - self.sample_size
                ret = np.concatenate((ret, self.p[:self.ind]), 0)
            self.reminder = self.reminder - 1
            return ret
        raise StopIteration()


class ReLUNetwork(nn.Module):
    """A basic fully-connected ReLU network."""

    def __init__(self, d, d_, k, T=1, kind="relu", bias=True, alpha=None,
        optimize_all=True):

        super(ReLUNetwork, self).__init__()
        self.d = d
        self.d_ = d_
        self.k = k
        self.T = T
        self.kind = kind
        self.activation = NON_LINEARITIES[self.kind]
        self.optimize_all = optimize_all

        self.layers = [nn.Linear(self.d, self.k, bias=bias)]
        for i in range(self.T - 1):
            self.layers.append(nn.Linear(self.k, self.k, bias=bias))
        self.readout = nn.Linear(self.k, self.d_, bias=bias)

        for i, l in enumerate(self.layers):
            self.add_module("l{}".format(i), l)
            nn.init.xavier_normal_(l.weight, nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.readout.weight)

    def forward(self, x):
        for l in self.layers:
            x = self.activation(l(x))
        return self.readout(x)

    def optimize(self, X, Y, criterion, batch_size, iters, lr,
        tqdm=lambda x: x):

        # if self.optimize_all:
        #     optimizer = Adam(self.parameters(), lr=lr)
        # else:
        #     optimizer = Adam(self.readout.parameters(), lr=lr)
        optimizer = SGD(self.parameters(), lr=lr)

        for ind in tqdm(Batches(X.shape[0], batch_size, iters)):
            optimizer.zero_grad()
            Y_hat = self(X[ind])
            loss = criterion(Y_hat, Y[ind])
            loss.backward()
            optimizer.step()

    def norm(self, ord):
        r = 0.
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                r += float(l.weight.norm(ord) ** ord)
            r += float(self.readout.weight.norm(ord) ** ord)
        return r



class GaLUNetwork(ReLUNetwork):
    """A basic fully-connected GaLU network."""

    @staticmethod
    def galu_filter(d, d_, alpha=1.):
        """
        Get a filter for a GaLU network, scaled by alpha.
        Note that scaling is not important when using the sign or gate
        non-linearities, but it is important for sigmoid/tanh.
        """
        return torch.randn(d, d_) / (alpha * np.sqrt(d))

    def __init__(self, d, d_, k, T=1, kind="gate", alpha=1.0, bias=True):
        super(GaLUNetwork, self).__init__(d, d_, k, T, kind, bias)
        self.filters = [GaLUNetwork.galu_filter(self.d, self.k, alpha)]
        for i in range(T - 1):
            self.filters.append(GaLUNetwork.galu_filter(self.k, self.k, alpha))

    def forward(self, x):
        for l, f in zip(self.layers, self.filters):
            x = l(x) * self.activation(x @ f) # Note: '*' means elementwise
                                              #       multiplication
        return self.readout(x)


class GaLU0Network(ReLUNetwork):
    """A basic fully-connected GaLU0 network."""

    def __init__(self, d, d_, k, T=1, kind="gate", alpha=1.0, bias=True):
        super(GaLU0Network, self).__init__(d, d_, k, T, kind, bias)
        self.filters = []
        a = alpha * np.sqrt(self.d)
        for i in range(T):
            self.filters.append(GaLUNetwork.galu_filter(self.d, self.k, a))

    def forward(self, x):
        x0 = x
        for l, f in zip(self.layers, self.filters):
            x = l(x) * self.activation(x0 @ f)
        return self.readout(x)



class GaLUNetwork_Shallow_R1(nn.Module):
    """A single layer GaLU network with output in $R^1$. It supports finding the
    optimial solution for the MSE loss, and a heurisitic hill-climbing
    algorithm to improve the gates."""

    def __init__(self, d, k, kind="gate"):
        super(GaLUNetwork_Shallow_R1, self).__init__()
        self.d = d
        self.k = k
        self.kind = kind
        self.activation = NON_LINEARITIES[self.kind]
        self.filters = torch.randn(self.d, self.k)
        self.linear = nn.Linear(self.d, self.k, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x) * self.activation(x @ self.filters)
        return x.sum(1, True)

    def optimize(self, X, Y):
        M = self.activation(X @ self.filters)
        X_bar = torch.cat([X * M[:, i, None] for i in range(self.k)], 1)
        w_star = (torch.pinverse(X_bar, 1e-5) @ Y).reshape(self.k, self.d)
        with torch.no_grad():
            self.linear.weight.copy_(w_star)

    def hill_climb_step(self, X, Y, current_loss):
        ind = np.random.randint(self.k)
        old_filter = self.filters[:, ind].clone()
        self.filters[:, ind] = torch.randn(self.d)
        self.optimize(X, Y)
        loss = ((self(X) - Y)**2).sum()
        if loss > current_loss:
            self.filters[:, ind] = old_filter
            return current_loss
        else:
            return loss

    def hill_climb(self, X, Y, iters, tqdm=lambda x: x):
        self.optimize(X, Y)
        if iters > 0:
            loss = ((self(X) - Y)**2).sum()
            for i in tqdm(range(iters)):
                loss = self.hill_climb_step(X, Y, loss)
            self.optimize(X, Y)

    def norm(self, ord):
        r = 0.
        with torch.no_grad():
            r += float(self.linear.weight.norm(ord) ** ord)
        return r



class GaLUNetwork_Shallow_Rd(nn.Module):
    """A single layer GaLU network with output in $R^d$. It supports Alternate-
    Convex-Search for the MSE loss: alternately finding the optimal solution
    for the first and second layers, keeping the other one constant. """

    def __init__(self, d, d_, k, kind="gate"):
        super(GaLUNetwork_Shallow_Rd, self).__init__()
        self.d = d
        self.d_ = d_
        self.k = k
        self.kind = kind
        self.activation = NON_LINEARITIES[self.kind]
        self.filters = torch.randn(self.d, self.k)
        self.A = nn.Linear(self.d, self.k, bias=False)
        self.B = nn.Linear(self.k, self.d_, bias=False)
        self.n_vars = self.d * self.k
        nn.init.xavier_normal_(self.A.weight)
        nn.init.xavier_normal_(self.B.weight)

    def forward(self, x):
        x = self.A(x) * self.activation(x @ self.filters)
        return self.B(x)

    def _optimize_init(self, X, Y):
        self.X = X
        self.Y = Y
        self.M = self.activation(self.X @ self.filters)
        self.X_ = torch.zeros((self.k, X.shape[0], self.d))
        for i in range(self.k):
            self.X_[i] = self.X * self.M[:, i:i+1]

        self.U_ = torch.zeros((self.k, self.k, self.d, self.d))
        self.R_ = torch.zeros((self.k, self.d, self.d_))
        for i in range(self.k):
            for j in range(self.k):
                self.U_[i, j] = self.X_[i].transpose(1, 0) @ self.X_[j]
            self.R_[i] = self.X_[i].transpose(1, 0) @ self.Y

    def _optimize_finalize(self):
        del self.X, self.Y, self.M, self.X_, self.U_, self.R_

    def optimize(self, X, Y, iters, tqdm=lambda x: x):
        self._optimize_init(X, Y)
        with torch.no_grad():
            for i in tqdm(range(iters)):
                self.update_A()
                self.update_B()
        self._optimize_finalize()

    def optimize_heuristic(self, X, Y, max_iters=1000000, eps=0.001):
        self._optimize_init(X, Y)
        loss, t, i = 2., 1, 0
        with torch.no_grad():
            while i <= max_iters:
                prev_loss = loss
                for j in range(2**t):
                    self.update_A()
                    self.update_B()
                    i += 1
                loss = ((self(X) - Y)**2).mean()
                if abs(prev_loss - loss) < eps * loss:
                    break
                t += 1
            else:
                print("Convergence Failed: prev_loss={:.2e} loss={:2e}".format(
                    prev_loss, loss))
        self._optimize_finalize()

    def update_B(self):
        b = torch.gels(self.Y, self.A(self.X) * self.M)[0]
        self.B.weight.copy_(b[:self.k].transpose(1, 0))

    def update_A(self):
        f = self.R_ @ self.B.weight.transpose(1, 0)[:, :, None]
        free = f.reshape(self.n_vars,)

        V = self.B.weight.transpose(1, 0) @ self.B.weight
        t = V[:, :, None, None] * self.U_
        coefs = t.permute(0, 2, 1, 3).reshape(self.n_vars, self.n_vars)

        a = torch.gesv(free, coefs)[0].reshape((self.k, self.d))
        self.A.weight.copy_(a)


NETWORKS = {
    "relu": ReLUNetwork,
    "relu_last": partial(ReLUNetwork, optimize_all=False),
    "sigmoid": partial(ReLUNetwork, kind="sigmoid"),
    "linear": partial(ReLUNetwork, kind="none"),
    "galu": GaLUNetwork,
    "galu0": GaLU0Network,
    "galus": partial(GaLUNetwork, kind="sigmoid"),
    "galu0s": partial(GaLU0Network, kind="sigmoid"),
}
