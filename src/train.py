import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import NoReturn
from tqdm import tqdm

# import internal libs
from utils import get_logger, set_seed

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


def extra_loss(device: torch.device,
               loss_type: int,
               features: torch.Tensor,
               seed: int) -> torch.Tensor:
    """extra loss
    
    Args:
        device: GPU or CPU
        loss_type: the loss type
        features: the features
        seed: the seed
    
    Returns:
        the extra loss
    """
    if loss_type == 1:
        return 0.0
    else:
        N, D = features.shape
        mean = features.mean(dim=0, keepdim=True)
        var = features.var(dim=0, unbiased=False, keepdim=True)
        y = (features - mean) / torch.sqrt(var)
        if loss_type == 2:
            set_seed(seed)
            g = torch.randn(D).double().to(device)
            return torch.matmul(y, g) # with shape of (N, )
        elif loss_type == 3:
            H_diag = torch.randn(1, D).double().to(device)
            return torch.bmm(y.view(N, 1, D), (H_diag * y).view(N, D, 1)).view(N)
        elif loss_type == 4:
            H_off = torch.randn(D, D).double().to(device)
            H_off.fill_diagonal_(0)
            return torch.bmm(torch.matmul(y, H_off).view(N, 1, D), y.view(N, D, 1)).view(N)
        else:
            raise ValueError(f"unknown loss type: {loss_type}")

def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          testset: Dataset,
          loss_type: int,
          batch_size: int,
          seed: int) -> NoReturn:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        testset: the test dataset
        loss_type: the loss type
        batch_size: the batch size
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
    test_batches = create_batches(testset, batch_size, seed)
    
    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    # add the hook things
    # forward one
    layers = {
        "classifier.0": model.classifier[0],
    }
    forward_hook_outputs = {}
    forward_handles = {}
    for layer_name, module in layers.items():
        def register_hook(module, layer_name):
            def hook(module, input, output):
                assert isinstance(output, torch.Tensor), f"output is not a tensor: {output}"
                forward_hook_outputs[layer_name] = output
            forward_handles[layer_name] = module.register_forward_hook(hook)
        register_hook(module, layer_name)
        
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

    # evaluatioin
    model.train()
    for batch_id, (inputs, labels) in enumerate(tqdm(test_batches)):
        # set the inputs to device
        inputs, labels = inputs.double().to(device), labels.to(device)
        # set the outputs
        outputs = model(inputs)
        # get the CE_loss
        CE_losses = loss_fn(outputs, labels)
        extra_losses = 0
        for layer_name, features in forward_hook_outputs.items():
            extra_losses += extra_loss(device, loss_type, features, seed)
        # set the loss
        loss = (CE_losses + extra_losses).sum(dim=0)
        # set the gradients to zero
        model.zero_grad()
        # backward
        loss.backward()
        # get the gradients
        for layer_name, grads in backward_hook_grads.items():
            logger.debug(f"{layer_name} gradients: {grads.shape}")
            torch.save(grads, os.path.join(save_path, f"{layer_name}_grad_batch{batch_id}.pt"))

    # remove the forward hook
    for handle in forward_handles.values():
        handle.remove()
    # remove the backward hook
    for handle in backward_handles.values():
        handle.remove()