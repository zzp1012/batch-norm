import os, sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

MODEL_NAME = "vgg11"
BN_TYPE = "custom"
DATASET = "cifar10"
MODEL_PATH = "NEED TO BE FILLED"
SAMPLE_NUM = 1000

sys.path.insert(1, os.path.join(os.path.dirname(MODEL_NAME), "../src/"))
# import internal libs
from data import prepare_dataset
from model import prepare_model

# load the dataset
dataset, _ = prepare_dataset(DATASET)
inputs, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))

# sample a set of inputs and labels with equal number for each classies
indices = []
for i, label in enumerate(range(len(dataset.classes))):
    indices_tmp = np.where(labels == label)[0]
    np.random.seed(seed=i)
    indices.extend(np.random.choice(indices_tmp, SAMPLE_NUM // len(dataset.classes), replace=False))
sampled_inputs, sampled_labels = inputs[indices], labels[indices].cpu().numpy()

# make the model
model = prepare_model(MODEL_NAME, DATASET, BN_TYPE)
# load the pretrained model
state_dict = torch.load(MODEL_PATH)
model.load_state_dict(state_dict)

# get the intermediate features
# define the hook
features = []
def hook(module, input, output):
    features.append(output.cpu().detach())
hook = model.classifier[0].register_forward_hook(hook)

# foward
model.eval()
with torch.no_grad():
    model(sampled_inputs)
features = features[0]

print(features[3][:10] / features[500][:10])
print(sampled_labels[3], sampled_labels[500])

print(torch.dot(features[3], features[500]) / torch.norm(features[3]) / torch.norm(features[500]))
print(features.shape)