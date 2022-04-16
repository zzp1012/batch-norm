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
            batch_indices.extend(np.array_split(indices, len(indices) // batch_size))
        random.Random(seed).shuffle(batch_indices)
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
          seed: int,
          method: str = "label") -> NoReturn:
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
        seed: the seed
    """
    logger = get_logger(__name__)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # put the model to GPU or CPU
    model = model.to(device)
    # set the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # set the loss function
    loss_fn = nn.CrossEntropyLoss()
    # set the epochs
    for epoch in range(epochs):
        logger.info(f"######Epoch - {epoch}")
        # create the batches for train
        train_batches = create_batches(trainset, batch_size, epoch + seed, method)
        # train the model
        model.train()
        for inputs, labels in tqdm(train_batches):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)
            # set the optimizer
            optimizer.zero_grad()
            # set the outputs
            outputs = model(inputs)
            # set the loss
            loss = loss_fn(outputs, labels)
            # set the loss
            loss.backward()
            # set the optimizer
            optimizer.step()