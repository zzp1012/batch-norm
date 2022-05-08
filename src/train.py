import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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


def extra_loss(device: torch.device,
               loss_type: int,
               X: torch.Tensor,
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
    N, D = X.shape
    mean = X.mean(dim=0, keepdim=True) # (1, D)
    var = X.var(dim=0, unbiased=False, keepdim=True) # (1, D)
    y = (X - mean) / torch.sqrt(var) # (N, D)
    if loss_type == 1:
        extra_losses = torch.zeros(N, device=device)
    elif loss_type == 2:
        np.random.seed(seed)
        g = np.random.randn(D)
        g = torch.from_numpy(g).double().to(device)
        extra_losses = torch.matmul(y, g) # with shape of (N, )
    elif loss_type == 3:
        np.random.seed(seed)
        H_diag = np.random.randn(1, D)
        H_diag = torch.from_numpy(H_diag).double().to(device)
        extra_losses = torch.bmm(y.view(N, 1, D), (H_diag * y).view(N, D, 1)).view(N) # with shape of (N, )
    elif loss_type == 4:
        np.random.seed(seed)
        H_off = np.random.randn(D, D)
        H_off = torch.from_numpy(H_off).double().to(device)
        H_off.fill_diagonal_(0)
        extra_losses = torch.bmm(torch.matmul(y, H_off).view(N, 1, D), y.view(N, D, 1)).view(N) # with shape of (N, )
    else:
        raise ValueError(f"unknown loss type: {loss_type}")
    return extra_losses


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
    
    # record the different loss thing and the grad thing
    res_dict = {
        "CE_loss": [],
        "extra_loss": [],
    }
    grad_dict = {}
    
    # start
    model.train()
    for batch_id, (inputs, labels) in enumerate(tqdm(test_batches)):
        # set the inputs to device
        inputs, labels = inputs.double().to(device), labels.to(device)
        # set the outputs
        outputs = model(inputs)
        # get the CE_loss
        CE_losses = loss_fn(outputs, labels)
        extra_losses = None
        for layer_name, features in forward_hook_outputs.items():
            if extra_losses is None:
                extra_losses = extra_loss(device, loss_type, features, seed)
            else:
                extra_losses += extra_loss(device, loss_type, features, seed)
        assert extra_losses is not None, f"extra_losses is None"
        # set the loss
        loss = (CE_losses + extra_losses).sum(dim=0)
        # set the gradients to zero
        model.zero_grad()
        # backward
        loss.backward()
        # get the gradients
        for layer_name, grads in backward_hook_grads.items():
            grads = grads.cpu()
            logger.debug(f"{layer_name} gradients: {grads.shape}")
            if layer_name not in grad_dict:
                grad_dict[layer_name] = [grads]
            else:
                grad_dict[layer_name].append(grads)

        # update the res_dict
        res_dict["CE_loss"].extend(CE_losses.detach().cpu().numpy())
        res_dict["extra_loss"].extend(extra_losses.detach().cpu().numpy())

    # record the results
    res_df = pd.DataFrame.from_dict(res_dict)
    res_df.to_csv(os.path.join(save_path, "loss.csv"), index = False)
    # the gradient ting
    for layer_name, grad_lst in grad_dict.items():
        # cat the gradients
        grad_lst = torch.cat(grad_lst, dim=0)
        # save the gradients
        torch.save(grad_lst, os.path.join(save_path, f"{layer_name}_grads.pt"))

    # remove the forward hook
    for handle in forward_handles.values():
        handle.remove()
    # remove the backward hook
    for handle in backward_handles.values():
        handle.remove()