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
                   seed: int,
                   method: str,) -> list:
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
    if method == "random":
        indices = np.arange(len(dataset))
        random.Random(seed).shuffle(indices)
        batch_indices = np.array_split(indices, len(dataset) // batch_size)
    elif method == "label":
        batch_indices = []
        repeat_num = 300
        for itr in range(1, repeat_num+1):
            for i, label in enumerate(range(len(dataset.classes))):    
                indices = np.where(labels == label)[0]
                random.Random(seed + itr + i).shuffle(indices)
                batch_indices.append(indices[:batch_size])
    else:
        raise ValueError(f"unknown method: {method}")
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
          method: str,
          seed: int) -> NoReturn:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        testset: the test dataset
        batch_size: the batch size
        method: the method to create batches
        seed: the seed
    """
    logger = get_logger(__name__)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # put the model to GPU or CPU
    model = model.double().to(device)
    # save the initial model
    torch.save(model.state_dict(), os.path.join(save_path, f"model_init.pt"))

    # set the batches
    test_batches = create_batches(testset, batch_size, seed, method)
    
    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    # add the hook things
    # forward one
    layers = {
        "classifier.0": model.classifier[0],
        "classifier.1": model.classifier[1],
        "classifier.2": model.classifier[2],
        "classifier.3": model.classifier[3],
        "classifier.4": model.classifier[4],
        "classifier.5": model.classifier[5],
        "classifier.6": model.classifier[6],
        "classifier.7": model.classifier[7],
    }
        
    # backward one
    backward_hook_grads = {}
    backward_handles = {}
    for layer_name, module in layers.items():
        def register_hook(module, layer_name):
            def get_grads(module, grad_input, grad_output):
                assert len(grad_output) == 1, f"the grad output length should be 1, but got {len(grad_output)}"
                backward_hook_grads[layer_name] = grad_output[0].clone().detach()
            backward_handles[layer_name] = module.register_backward_hook(get_grads)
        register_hook(module, layer_name)
    
    # start
    model.train()
    for batch_id, (inputs, labels) in enumerate(tqdm(test_batches)):
        # set the inputs to device
        inputs, labels = inputs.double().to(device), labels.to(device)
        print(labels.unique())
        # set the outputs
        outputs = model(inputs)
        # get the CE_loss
        CE_losses = loss_fn(outputs, labels)
        # set the loss
        loss = CE_losses.sum(dim=0)
        # set the gradients to zero
        model.zero_grad()
        # backward
        loss.backward()
        # get the gradients
        for layer_name, grads in backward_hook_grads.items():
            grads = grads.cpu()
            logger.debug(f"#########{layer_name} gradients: {grads.shape}")
            logger.info(f"gradient norm: {torch.norm(grads) ** 2 / grads.numel()}")
        break

    # remove the backward hook
    for handle in backward_handles.values():
        handle.remove()