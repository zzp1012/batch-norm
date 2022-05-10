import os, sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader

MODEL_NAME = "vgg11"
BN_TYPE = "normal"
DATASET = "cifar10"
MODEL_PATH = "/data2/zzp1012/batch-norm/outs/tmp/0510-132638-seed0-cifar10-vgg11-bn_normal-label-epochs100-lr0.01-bs128-wd0.0-momentum0.0/exp/model_40.pt"
DATA_PATH = "/data2/zzp1012/batch-norm/data"
DEVICE = torch.device("cuda:0")
SAMPLE_NUM = 1000
BATCH_SIZE = 64
SEED = 0

sys.path.insert(1, os.path.join(os.path.dirname(MODEL_PATH), "../src/"))
# import internal libs
from data import prepare_dataset
from model import prepare_model

# load the dataset
dataset, _ = prepare_dataset(DATASET, DATA_PATH)
inputs, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))

# make the model
model = prepare_model(MODEL_NAME, DATASET, BN_TYPE)
# load the pretrained model
state_dict = torch.load(MODEL_PATH)
model.load_state_dict(state_dict)
model.to(DEVICE)

# sample a set of inputs and labels with equal number for each classies
indices = []
for i, label in enumerate(range(len(dataset.classes))):
    if i != 3:
        continue
    indices_selected = np.where(labels == label)[0]
    np.random.seed(seed=i)
    indices.extend(np.random.choice(indices_selected, 
        SAMPLE_NUM // len(dataset.classes), replace=False))
indices = np.array(indices)
# create the batches
random.Random(SEED).shuffle(indices)
batch_indices = np.array_split(indices, len(indices) // BATCH_SIZE)

features = []
def hook(module, input, output):
    features.append(output.cpu().detach().numpy())
handle = model.classifier[0].register_forward_hook(hook)

# foward
model.eval()
with torch.no_grad():
    for idx in batch_indices:
        sampled_inputs = inputs[idx].to(DEVICE)
        sampled_labels = labels[idx].to(DEVICE)

        outputs = model(sampled_inputs)
        # calculate the probability
        prob = torch.softmax(outputs, dim=1).max(dim=1)[0]
        # calculate the predicted label
        predicted = torch.max(outputs, 1)[1]
        
        feature = features[-1]
        del features[-1]

        print(feature.mean(axis=0))
        print(model.classifier[1].running_mean)

        for i in range(len(idx)):
            print(f"{sampled_labels[i]} {predicted[i]} {prob[i]}")

handle.remove()