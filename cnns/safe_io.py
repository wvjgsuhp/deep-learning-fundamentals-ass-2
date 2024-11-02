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


def read_cifar10_test() -> tuple[NPFloatMatrix, NPInt, list[str]]:
    test_file = "cifar-10-batches-py/test_batch"
    with open(test_file, "rb") as f:
        raw_test_set = pickle.load(f, encoding="bytes")

    # transform to a list of images with pixel values
    x_test = np.transpose(raw_test_set[b"data"].reshape((10000, 3, -1)), axes=[0, 2, 1]).reshape(
        (-1, 32, 32, 3)
    )
    y_test = np.array(raw_test_set[b"labels"])

    label_file = "cifar-10-batches-py/batches.meta"
    with open(label_file, "rb") as f:
        raw_labels = pickle.load(f, encoding="bytes")

    label_names = [raw.decode() for raw in raw_labels[b"label_names"]]

    return x_test, y_test, label_names


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
