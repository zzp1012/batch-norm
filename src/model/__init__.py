from ast import Num
import torch.nn as nn

# import internal libs
from utils import get_logger

def prepare_model(model_name: str,
                  dataset: str) -> nn.Module:
    """prepare the random initialized model according to the name.

    Args:
        model_name (str): the model name
        dataset (str): the dataset name

    Return:
        the model
    """
    logger = get_logger(__name__)
    logger.info(f"prepare the {model_name} model for dataset {dataset}")
    if model_name.startswith("vgg") and dataset.startswith("cifar"):
        import model.cifar_vgg as cifar_vgg
        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        else:
            raise ValueError(f"{dataset} is not supported.")
        model = cifar_vgg.__dict__[model_name](num_classes=num_classes)
    else:
        raise ValueError(f"unknown model name: {model_name}")
    return model
