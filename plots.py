from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

import parse

plt.style.use("ggplot")



################################################################################
#                                                                              #
#                     Figure 1: Datasets                                       #
#                                                                              #
################################################################################


def plot_networks(r, select, x_name, y_name):
    select["network_type"] = "relu"
    x, y = r.plot_results(select, x_name, y_name)
    plt.plot(x, y, "-o", label="ReLU")
    select["network_type"] = "galu0s"
    x, y = r.plot_results(select, x_name, y_name)
    plt.plot(x, y, "-x", label="GaLU0")
    select["network_type"] = "linear"
    x, y = r.plot_results(select, x_name, y_name)
    plt.plot(x, y, "-1", label="Linear")
    return x

r = parse.Results(["results/datasets.txt"])
r.parse_datasets()
r.best_lr_a("test_01")

plt.figure(figsize = (8, 4))

plt.subplot(1, 2, 1)
x = plot_networks(r, {"dataset": "mnist"}, "k", "test_01")
plt.title("MNIST")
plt.xlabel(r"$k$")
plt.ylabel("1 - Test Accuracy")
plt.ylim(0., 0.1)
plt.yticks(np.arange(0, 11) / 100)
plt.xticks(x)

plt.subplot(1, 2, 2)
x = plot_networks(r, {"dataset": "fmnist"}, "k", "test_01")
plt.legend(loc="upper right")
plt.title("Fashion MNIST")
plt.xlabel(r"$k$")
plt.ylabel("1 - Test Accuracy")
plt.ylim(0.1, 0.2)
plt.yticks(np.arange(10, 21) / 100)
plt.xticks(x)

plt.subplots_adjust(wspace=0.3)
tikz_save("plots/datasets.tex")
plt.savefig("plots/datasets.png")
plt.clf()



################################################################################
#                                                                              #
#                     Figure 2: R^1 random regression                          #
#                                                                              #
################################################################################


plt.figure(figsize = (6, 5))

def avg_same_x(X, Y):
    d = defaultdict(list)
    for x, y in zip(X, Y):
        d[x].append(y)
    X = np.zeros(len(d))
    Y = np.zeros(len(d))
    for i, x in enumerate(sorted(d.keys())):
        X[i] = x
        Y[i] = np.mean(d[x])
    return X, Y

r = parse.Results("results/r1.txt")
r.parse_memorization()


f = lambda h: h["k"] * h["d"] / h["m"]
X, Y = avg_same_x(*r.plot_results({"network_type": "relu"}, f, "loss"))
plt.plot(X, Y, "-o", label="ReLU")

X, Y = avg_same_x(*r.plot_results({"network_type": "galu_opt", "hill_climb_iters": 0}, f, "loss"))
plt.plot(X, Y, "-x", label="GaLU")

X, Y = avg_same_x(*r.plot_results({"network_type": "galu_opt", "hill_climb_iters": 512}, f, "loss"))
plt.plot(X, Y, "-1", label="GaLU(512)")


X = np.linspace(0, 1, 10)
Y = 1 - X
plt.plot(X, Y, "--", label=r"$1-\frac{kd}{m}$", linewidth=7., alpha=0.5)
X = np.linspace(0, 0.5, 10)
Y = 1 - 2 * X
plt.plot(X, Y, "--", label=r"$1-2\frac{kd}{m}$", linewidth=7., alpha=0.5)

plt.xlabel(r"$\frac{kd}{m}$")
plt.ylabel(r"MSE")
plt.legend()

tikz_save("plots/r1_regression.tex")
plt.savefig("plots/r1_regression.png")
plt.clf()



################################################################################
#                                                                              #
#                     Figure 3: Train and Test Errors                          #
#                                                                              #
################################################################################


r = parse.Results(["results/generalization.txt"])
r.parse_generalization()

plt.figure(figsize = (12, 4))

plt.subplot(1, 3, 1)

x, y = r.plot_results({"network_type": "relu", "problem_type": "regression"}, "sigma_y", "train_ce")
plt.plot(x, y, "-o", label="ReLU Train")
x, y = r.plot_results({"network_type": "relu", "problem_type": "regression"}, "sigma_y", "test_ce")
plt.plot(x, y, "-o", label="ReLU Test")
x, y = r.plot_results({"network_type": "galu", "problem_type": "regression"}, "sigma_y", "train_ce")
plt.plot(x, y, "-o", label="GaLU Train")
x, y = r.plot_results({"network_type": "galu", "problem_type": "regression"}, "sigma_y", "test_ce")
plt.plot(x, y, "-o", label="GaLU Test")

X = np.linspace(0, 1, 10)
plt.plot(X, X**2, "--", label="Optimum", linewidth=7., alpha=0.5)


plt.title("Regression")
plt.xlabel(r"$\sigma_y$")
plt.ylabel("MSE")
plt.legend(loc="upper left")

plt.subplot(1, 3, 2)
x, y = r.plot_results({"network_type": "relu", "problem_type": "classification"}, "sigma_y", "train_ce")
plt.plot(x / 2, y, "-o", label="ReLU Train")
x, y = r.plot_results({"network_type": "relu", "problem_type": "classification"}, "sigma_y", "test_ce")
plt.plot(x / 2, y, "-o", label="ReLU Test")
x, y = r.plot_results({"network_type": "galu", "problem_type": "classification"}, "sigma_y", "train_ce")
plt.plot(x / 2, y, "-o", label="GaLU Train")
x, y = r.plot_results({"network_type": "galu", "problem_type": "classification"}, "sigma_y", "test_ce")
plt.plot(x / 2, y, "-o", label="GaLU Test")


plt.title("Classification")
plt.xlabel(r"$p$")
plt.ylabel("Hinge Loss")


plt.subplot(1, 3, 3)
x, y = r.plot_results({"network_type": "relu", "problem_type": "classification"}, "sigma_y", "train_01")
plt.plot(x / 2, y, "-o", label="ReLU Train")
x, y = r.plot_results({"network_type": "relu", "problem_type": "classification"}, "sigma_y", "test_01")
plt.plot(x / 2, y, "-o", label="ReLU Test")
x, y = r.plot_results({"network_type": "galu", "problem_type": "classification"}, "sigma_y", "train_01")
plt.plot(x / 2, y, "-o", label="GaLU Train")
x, y = r.plot_results({"network_type": "galu", "problem_type": "classification"}, "sigma_y", "test_01")
plt.plot(x / 2, y, "-o", label="GaLU Test")

X = np.linspace(0, 0.5, 10)
plt.plot(X, X, "--", label="Optimum", linewidth=7., alpha=0.5)

plt.title("Classification")
plt.xlabel(r"$p$")
plt.ylabel("Accuracy")

plt.subplots_adjust(wspace=0.3)
tikz_save("plots/generalization.tex")
plt.savefig("plots/generalization.png")
plt.clf()



################################################################################
#                                                                              #
#                  Figure 4: Generalization Error and norms                    #
#                                                                              #
################################################################################


plt.figure(figsize = (12, 8))

_, galu_train_loss_reg = r.plot_results({"network_type": "galu_opt", "problem_type": "regression"}, "sigma_y", "train_ce")
_, galu_test_loss_reg = r.plot_results({"network_type": "galu_opt", "problem_type": "regression"}, "sigma_y", "test_ce")
_, galu_l1_reg = r.plot_results({"network_type": "galu_opt", "problem_type": "regression"}, "sigma_y", "l1")
_, galu_l2_reg = r.plot_results({"network_type": "galu_opt", "problem_type": "regression"}, "sigma_y", "l2")

plt.subplot(1, 2, 1)
plt.plot(galu_l2_reg, galu_test_loss_reg - galu_train_loss_reg, "o-", label="GaLU MSE Loss")
plt.xlabel("$\ell_2^2$ of weights")
plt.ylabel("test loss - train loss")
plt.title("$\ell_2$")

plt.subplot(1, 2, 2)
plt.plot(galu_l1_reg ** 2, galu_test_loss_reg - galu_train_loss_reg, "o-", label="GaLU MSE Loss")
plt.xlabel("$\ell_1^2$ of weights")
plt.ylabel("test loss - train loss")
plt.title("$\ell_2$")

plt.subplots_adjust(wspace=0.3, hspace=0.3)
tikz_save("plots/generalization_norms.tex")
plt.savefig("plots/generalization_norms.png")
plt.clf()




################################################################################
#                                                                              #
#                  Figure 5: R^d random regression                             #
#                                                                              #
################################################################################


plt.figure(figsize = (6, 5))
r = parse.Results("results/minimal_k.txt")
r.parse_memorization()

def minimal_k(r, network_type):
    d = defaultdict(list)
    for header, data in r.select({"network_type": network_type}):
        if data["loss"] < 0.3:
            d[header["d_"]].append(header["k"])
    D_ = np.zeros(len(d))
    K = np.zeros(len(d))
    for i, d_ in enumerate(sorted(d.keys())):
        D_[i] = d_
        K[i] = np.min(d[d_])
    return D_, K

D_, K = minimal_k(r, "relu")
plt.plot(D_, K, "-o", label="ReLU")
D_, K = minimal_k(r, "galu0")
plt.plot(D_, K, "-x", label="GaLU")
plt.xlabel(r"Output Dimension $d'$")
plt.ylabel("Minimal $k$")
plt.xticks(D_)
plt.legend()
plt.title(r"Minimal Number of Neurons to achieve MSE$<0.3$")

tikz_save("plots/minimal_k.tex")
plt.savefig("plots/minimal_k.png")
plt.clf()
