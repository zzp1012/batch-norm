import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import NoReturn
from tqdm import tqdm

# import internal libs
from utils import get_logger

def create_batches(dataset: Dataset,
                   batch_size: int,
                   seed: int) -> list:
    """create the batches

    Args:
        dataset: the dataset
        batch_size: the batch size
        seed: the seed
        method: the method to create batches

    Return:
        the batches
    """
    logger = get_logger(f"{__name__}.create_batches")
    # use dataloader
    inputs, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    logger.debug(f"inputs shape: {inputs.shape}; labels shape: {labels.shape}")
    # create the indices
    indices = np.arange(len(dataset))
    random.Random(seed).shuffle(indices)
    batch_indices = np.array_split(indices, len(dataset) // batch_size)
    # create the batches
    batches = []
    for idx in batch_indices:
        batches.append((inputs[idx], labels[idx]))
    return batches


def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          testset: Dataset,
          batch_size: int,
          seed: int) -> NoReturn:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        testset: the test dataset
        batch_size: the batch size
        seed: the seed
    """
    logger = get_logger(__name__)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # put the model to GPU or CPU
    model = model.to(device)
    # save the initial model
    torch.save(model.state_dict(), os.path.join(save_path, f"model_init.pt"))

    # set the batches
    test_batches = create_batches(testset, batch_size, seed)
    
    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    # add the hook things
    bn_layers = {
        "classifier.1": model.classifier[1],
    }
    forward_hook_inputs = {}
    forward_handles = {}
    for layer_name, module in bn_layers.items():
        def register_hook(module, layer_name):
            def hook(module, input, output):
                assert len(input) == 1, f"the input length should be 1, but got {len(input)}"
                forward_hook_inputs[layer_name] = input[0]
            forward_handles[layer_name] = module.register_forward_hook(hook)
        register_hook(module, layer_name)
        
    # evaluatioin
    model.train()
    for inputs, labels in tqdm(test_batches):
        # set the inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        # set the outputs
        outputs = model(inputs)
        # set the loss
        CE_losses = loss_fn(outputs, labels)


    # remove the forward hook
    for handle in forward_handles.values():
        handle.remove()