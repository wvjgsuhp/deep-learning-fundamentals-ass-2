import hashlib
import os.path
import pickle
import sys
import tarfile
import urllib.request

import numpy as np

from .custom_types import NPFloatMatrix, NPInt


def download_cifar10(cifar10_file: str = "cifar-10-python.tar.gz") -> None:
    if cifar10_exists(cifar10_file):
        return

    print(f"downloading {cifar10_file}")
    urllib.request.urlretrieve(
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        cifar10_file,
        reporthook,
    )


def extract_cifar10(cifar10_file: str) -> None:
    print(f"extracting {cifar10_file}")
    with tarfile.open(cifar10_file, "r:gz") as tar:
        tar.extractall()


def read_cifar10_test() -> tuple[NPFloatMatrix, NPInt]:
    return _read_cifar10("cifar-10-batches-py/test_batch")


def read_cifar10_train() -> tuple[NPFloatMatrix, NPInt]:
    base_file = "cifar-10-batches-py/data_batch_"
    x, y = _read_cifar10(f"{base_file}1")

    # combine test datasets
    for i in range(2, 6):
        x_i, y_i = _read_cifar10(f"{base_file}{i}")
        x = np.append(x, x_i, axis=0)
        y = np.append(y, y_i, axis=0)

    return x, y


def _read_cifar10(cifar10_file: str) -> tuple[NPFloatMatrix, NPInt]:
    with open(cifar10_file, "rb") as f:
        raw = pickle.load(f, encoding="bytes")

    # transform to a list of images (matrix of RGB)
    x = np.transpose(raw[b"data"].reshape((raw[b"data"].shape[0], 3, -1)), axes=[0, 2, 1]).reshape(
        (-1, 32, 32, 3)
    )
    y = np.array(raw[b"labels"])

    return x, y


def read_cifar10_labels() -> list[str]:
    label_file = "cifar-10-batches-py/batches.meta"
    with open(label_file, "rb") as f:
        raw_labels = pickle.load(f, encoding="bytes")

    return [raw.decode() for raw in raw_labels[b"label_names"]]


# https://stackoverflow.com/a/13895723
def reporthook(blocknum: float, blocksize: float, totalsize: float) -> None:
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent,
            len(str(totalsize)),
            readsofar,
            totalsize,
        )
        sys.stderr.write(s)
        if readsofar >= totalsize:
            sys.stderr.write("\n")
    else:
        sys.stderr.write("read %d\n" % (readsofar,))


def cifar10_exists(cifar10_file: str) -> bool:
    return (
        # check if file exists
        os.path.isfile(cifar10_file)
        # check if file contents match
        and hashlib.md5(open(cifar10_file, "rb").read()).hexdigest() == "c58f30108f718f92721af3b95e74349a"
    )
