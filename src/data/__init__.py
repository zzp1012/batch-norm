import torch
from torch.utils.data import Dataset
from typing import Tuple

def prepare_dataset(dataset: str,
                    root: str = "../data/") -> Tuple[Dataset, Dataset]:
    """prepare the dataset.

    Args:
        dataset (str): the dataset name.
        root (str): the root path of the dataset.

    Returns:
        trainset and testset
    """
    if dataset == "mnist":
        import data.mnist as mnist
        trainset, testset = mnist.load(root)
    else:
        raise NotImplementedError(f"dataset {dataset} is not implemented.")
    return trainset, testset