import torch
import torch.nn as nn

def prepare_model(model_name: str) -> nn.Module:
    """prepare the random initialized model according to the name.

    Args:
        model_name (str): the model name

    Return:
        the model
    """
    if model_name.startswith("VGG"):
        from model.vgg import VGG
        return VGG(model_name)
    else:
        raise ValueError(f"unknown model name: {model_name}")
