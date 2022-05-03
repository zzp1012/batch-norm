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
from utils import get_logger, update_dict

def create_batches(dataset: Dataset,
                   batch_size: int,
                   seed: int,
                   method: str) -> list:
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
        for i, label in enumerate(range(len(dataset.classes))):
            indices = np.where(labels == label)[0]
            random.Random(seed + i).shuffle(indices)
            batch_indices.append(np.array_split(indices, len(indices) // batch_size))
        batch_indices = np.array(batch_indices)
        batch_indices = batch_indices.T.ravel()
    # create the batches
    batches = []
    for idx in batch_indices:
        batches.append((inputs[idx], labels[idx]))
    return batches


def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          epochs: int,
          lr: float,
          batch_size: int,
          weight_decay: float,
          momentum: float,
          seed: int,
          method: str = "random") -> NoReturn:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        trainset: the train dataset
        testset: the test dataset
        epochs: the epochs number
        lr: the learning rate
        batch_size: the batch size
        weight_decay: the weight decay
        momentum: the momentum
        seed: the seed
    """
    logger = get_logger(__name__)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # put the model to GPU or CPU
    model = model.to(device)
    # set the optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=lr, weight_decay=weight_decay, momentum=momentum)
    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    # initialize the res_dict
    total_res_dict = {
        "train_loss": [],
        "train_ridge_loss": [],
        "train_acc": [],
        "cosine": [],
        "test_loss": [],
        "test_acc": [],
    }
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
                forward_hook_inputs[layer_name] = input[0].clone().detach()
            forward_handles[layer_name] = module.register_forward_hook(hook)
        register_hook(module, layer_name)
    # save the initial model
    torch.save(model.state_dict(), os.path.join(save_path, f"model_init.pt"))
    for epoch in range(1, epochs+1):
        logger.info(f"######Epoch - {epoch}")
        # create the batches for train
        train_batches = create_batches(trainset, batch_size, epoch + seed, method)
        # train the model
        model.train()
        train_losses, train_ridge_losses, train_acc = [], [], 0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_batches)):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)
            # set the outputs
            outputs = model(inputs)
            # set the ridge losses
            ridge_losses = torch.norm(forward_hook_inputs["classifier.1"], p='fro', dim=(-1)) ** 2 
            # set the loss
            losses = loss_fn(outputs, labels)
            loss = torch.mean(losses)
            # set zero grad
            optimizer.zero_grad()
            # set the loss
            loss.backward()
            # set the optimizer
            optimizer.step()    
            # set the loss and accuracy
            train_losses.extend(losses.cpu().detach().numpy())
            train_ridge_losses.extend(ridge_losses.cpu().detach().numpy())
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            # save the bn running_mean and running_var
            for name, module in bn_layers.items():
                bn_param_path = os.path.join(save_path, "bn_params", f"{name}")
                os.makedirs(bn_param_path, exist_ok=True)
                try:
                    torch.save(module.running_mean, os.path.join(bn_param_path, f"epoch{epoch}_batch{batch_idx}_mean.pth"))
                    torch.save(module.running_var, os.path.join(bn_param_path, f"epoch{epoch}_batch{batch_idx}_var.pth"))
                except:
                    bn_layers = {}
                    break
        # print the train loss and accuracy
        train_loss = np.mean(train_losses)
        train_ridge_loss = np.mean(train_ridge_losses)
        train_acc /= len(trainset)
        logger.info(f"train loss: {train_loss}; ridge loss: {train_ridge_loss}; train accuracy: {train_acc}")

        # evaluatioin
        model.eval()
        with torch.no_grad():
            # calculate the cosine
            inputs, labels = next(iter(DataLoader(testset, batch_size=batch_size, shuffle=False)))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            features = forward_hook_inputs["classifier.1"]
            assert len(features.shape) == 2, "the features should be a 2-D tensor"
            cosine = torch.cosine_similarity(features[:-1], features[1:], dim = 1).mean().item()
            logger.info("cosine: {}".format(cosine))

            # testset
            test_losses, test_acc = [], 0
            testloader = DataLoader(testset, batch_size=batch_size)
            for inputs, labels in tqdm(testloader):
                # set the inputs to device
                inputs, labels = inputs.to(device), labels.to(device)
                # set the outputs
                outputs = model(inputs)
                # set the loss
                losses = loss_fn(outputs, labels)
                # set the loss and accuracy
                test_losses.extend(losses.cpu().detach().numpy())
                test_acc += (outputs.max(1)[1] == labels).sum().item()
        # print the test loss and accuracy
        test_loss = np.mean(test_losses)
        test_acc /= len(testset)
        logger.info(f"test loss: {test_loss}; test accuracy: {test_acc}")

        # update res_dict
        res_dict = {
            "train_loss": [train_loss],
            "train_ridge_loss": [train_ridge_loss],
            "train_acc": [train_acc],
            "cosine": [cosine],
            "test_loss": [test_loss],
            "test_acc": [test_acc],
        }
        total_res_dict = update_dict(res_dict, total_res_dict)

        # save the results
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(), 
                       os.path.join(save_path, f"model_{epoch}.pt"))
            res_df = pd.DataFrame.from_dict(total_res_dict)
            res_df.to_csv(os.path.join(save_path, "train.csv"), index = False)

    # remove the forward hook
    for handle in forward_handles.values():
        handle.remove()