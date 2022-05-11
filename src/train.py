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
                   pos_lbl: int,
                   neg_lbl: int,) -> list:
    """create the batches

    Args:
        dataset: the dataset
        batch_size: the batch size
        seed: the seed
        pos_lbl: the label of positive samples.
        neg_lbl: the label of negative samples.

    Return:
        the batches
    """
    logger = get_logger(f"{__name__}.create_batches")
    # use dataloader
    inputs, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    logger.debug(f"inputs shape: {inputs.shape}; labels shape: {labels.shape}")
    # create the indices
    indices = np.where((labels == neg_lbl) | (labels == pos_lbl))[0]
    random.Random(seed).shuffle(indices)
    batch_indices = np.array_split(indices, len(indices) // batch_size)
    # make pos lablels to be 1
    labels[labels == pos_lbl] = 1
    # make neg lablels to be 0
    labels[labels == neg_lbl] = 0
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
          pos_lbl: int,
          neg_lbl: int,
          epochs: int,
          lr: float,
          batch_size: int,
          weight_decay: float,
          momentum: float,
          seed: int) -> NoReturn:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        trainset: the train dataset
        testset: the test dataset
        pos_lbl: the label of positive samples.
        neg_lbl: the label of negative samples.
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
    # create test batches
    test_batches = create_batches(testset, batch_size, seed, pos_lbl, neg_lbl)
    # set the loss function
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    # initialize the res_dict
    total_res_dict = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    # save the initial model
    torch.save(model.state_dict(), os.path.join(save_path, f"model_init.pt"))
    for epoch in range(1, epochs+1):
        logger.info(f"######Epoch - {epoch}")
        # train the model
        model.train()
        train_losses, train_acc = [], 0
        # create the batches for train
        train_batches = create_batches(trainset, batch_size, epoch + seed, pos_lbl, neg_lbl)
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_batches)):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.float().to(device)
            # set the outputs
            logits = model(inputs) # (N, 1)
            logits = logits.view(logits.shape[0]) # (N,)
            # set the loss
            losses = loss_fn(logits, labels)
            loss = torch.mean(losses)
            # set zero grad
            optimizer.zero_grad()
            # set the loss
            loss.backward()
            # set the optimizer
            optimizer.step()    
            # set the loss and accuracy
            train_losses.extend(losses.cpu().detach().numpy())
            train_acc += (torch.round(torch.sigmoid(logits)) == labels).sum().item() # torch.round(0.5) = 0

        # print the train loss and accuracy
        train_loss = np.mean(train_losses)
        train_acc /= len(train_losses)
        logger.info(f"train loss: {train_loss}; train accuracy: {train_acc}")

        # evaluatioin
        model.eval()
        with torch.no_grad():
            # testset
            test_losses, test_acc = [], 0
            for inputs, labels in tqdm(test_batches):
                # set the inputs to device
                inputs, labels = inputs.to(device), labels.float().to(device)
                # set the outputs
                logits = model(inputs)
                logits = logits.view(logits.shape[0])
                # set the loss
                losses = loss_fn(logits, labels)
                # set the loss and accuracy
                test_losses.extend(losses.cpu().detach().numpy())
                test_acc += (torch.round(torch.sigmoid(logits)) == labels).sum().item() # torch.round(0.5) = 0
        # print the test loss and accuracy
        test_loss = np.mean(test_losses)
        test_acc /= len(test_losses)
        logger.info(f"test loss: {test_loss}; test accuracy: {test_acc}")

        # update res_dict
        res_dict = {
            "train_loss": [train_loss],
            "train_acc": [train_acc],
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