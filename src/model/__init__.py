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
    if dataset == "mnist":
        if model_name == "AlexNet":
            from model.alexnet import AlexNet
            model = AlexNet()
        elif model_name == "LeNet5":
            from model.lenet5 import LeNet5
            model = LeNet5()
        else:
            raise ValueError(f"unknown model name: {model_name} for dataset {dataset}")
    else:
        raise ValueError(f"{dataset} is not supported.")
    return model
