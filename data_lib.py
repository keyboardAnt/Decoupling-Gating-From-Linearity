import torch
from torchvision import datasets

DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
    "cifar": datasets.CIFAR10
}


NUM_CHANNELS = {
    "mnist": 1,
    "fmnist": 1,
    "cifar": 3,
}


def rp_dataset(dataset, d):
    """Load the dataset and reduce the images dimension using a random linear
    projection."""

    train = DATASETS[dataset]("datasets/{}/".format(dataset), train=True,
        download=True)
    train_d = train.train_data.reshape((train.train_data.shape[0], -1)).to(
        torch.float32)

    means = train_d.mean(0, True)
    train_d -= means
    stds = train_d.std(0, True)
    stds[stds == 0.] = 1.
    train_d /= stds

    R = torch.randn(train_d.shape[1], d) / np.sqrt(train_d.shape[1])
    X_train = train_d @ R
    Y_train = train.train_labels

    test = DATASETS[dataset]("datasets/{}/".format(dataset), train=False,
        download=True)
    test_d = test.test_data.reshape((test.test_data.shape[0], -1)).to(
        torch.float32)
    test_d = (test_d - means) / stds

    X_test = test_d @ R
    Y_test = test.test_labels

    return (X_train, Y_train), (X_test, Y_test)


def pca_dataset(dataset, d):
    """Load the dataset and reduce the images dimension using PCA."""

    train = DATASETS[dataset]("datasets/{}/".format(dataset), train=True,
        download=True)
    train_d = train.train_data.reshape((train.train_data.shape[0], -1)).to(
        torch.float32)

    test = DATASETS[dataset]("datasets/{}/".format(dataset), train=False,
        download=True)
    test_d = test.test_data.reshape((test.test_data.shape[0], -1)).to(
        torch.float32)

    means = train_d.mean(0, True)
    zero_mean = train_d - means
    U, S, V = torch.svd(zero_mean.t() @ zero_mean)
    R = U[:, :d]
    stds = (zero_mean @ R).std(0, True)

    X_train = ((train_d - means) @ R) / stds
    Y_train = train.train_labels

    X_test = ((test_d - means) @ R) / stds
    Y_test = test.test_labels

    return (X_train, Y_train), (X_test, Y_test)


def simple_dataset(dataset):
    """Load the dataset."""

    train = DATASETS[dataset]("datasets/{}/".format(dataset), train=True,
        download=True)
    train_d = train.train_data.to(torch.float32)

    test = DATASETS[dataset]("datasets/{}/".format(dataset), train=False,
        download=True)
    test_d = test.test_data.to(torch.float32)

    means = train_d.mean(0, True)
    zero_mean = train_d - means
    stds = zero_mean.std(0, True)
    stds[stds == 0] = 1.

    X_train = (train_d - means) / stds
    Y_train = train.train_labels

    X_test = (test_d - means) / stds
    Y_test = test.test_labels

    if X_train.dim() == 3:
        return (X_train[:, None,:, :], Y_train), (X_test[:, None,:, :], Y_test)
    else:
        return (X_train, Y_train), (X_test, Y_test)


def loader_dataset(dataset, transform, batch_size):
    """Return a torchvision DataLoader object."""

    trainset = DATASETS[dataset]("datasets/{}/".format(dataset), train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

    testset = DATASETS[dataset]("datasets/{}/".format(dataset), train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

    return trainloader, testloader


def batches(sample_size, batch_size, iters):
    p = np.random.permutation(self.sample_size)
    ind = 0
    for i in range(iters):
        end = ind + batch_size
        if end < sample_size:
            yield p[ind : end]
        else:
            end = end - sample_size
            q = np.random.permutation(self.sample_size)
            yield np.concatenate((p[ind:], q[:end]))
            p = q
        ind = end


def single_epoch(sample_size, batch_size, last=False):
    end = sample_size if last else (sample_size - batch_size + 1)
    p = np.random.permutation(self.sample_size)
    for i in range(0, end, batch_size):
        yield p[i : i + batch_size]
