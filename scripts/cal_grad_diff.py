import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# set the global path
loss1_grad_path = "/lustre/home/acct-eezqs/eezqs/zzp1012/batch-norm-exp2/outs/0511/loss1/0511-144721-seed0-cifar10-vgg11-loss1-bs128/exp2"
loss2_grad_path = "/lustre/home/acct-eezqs/eezqs/zzp1012/batch-norm-exp2/outs/0511/loss2/0511-144807-seed0-cifar10-vgg11-loss2-bs128/exp2"
loss3_grad_path = "/lustre/home/acct-eezqs/eezqs/zzp1012/batch-norm-exp2/outs/0511/loss3/0511-144854-seed0-cifar10-vgg11-loss3-bs128/exp2"
loss4_grad_path = "/lustre/home/acct-eezqs/eezqs/zzp1012/batch-norm-exp2/outs/0511/loss4/0511-144941-seed0-cifar10-vgg11-loss4-bs128/exp2"
SAMPLE_NUM = 10000
BATCH_SIZE = 128


# define the dict to store the paths
grad_path_dict = {
    "loss1": loss1_grad_path,
    "loss2": loss2_grad_path,
    "loss3": loss3_grad_path,
    "loss4": loss4_grad_path,
}

# create a table
data_mean, data_std = [], []
for loss_type_1, grad_path_1 in grad_path_dict.items():
    print(f"row: {loss_type_1}")
    row_mean, row_std = [], []
    for loss_type_2, grad_path_2 in grad_path_dict.items():
        print(f"col: {loss_type_2}")
        
        grad_diff_lst = []
        for i in tqdm(range(SAMPLE_NUM // BATCH_SIZE)):
            grad_1 = torch.load(os.path.join(grad_path_1, f"classifier.0_grads_bs{i}.pt"))
            grad_2 = torch.load(os.path.join(grad_path_2, f"classifier.0_grads_bs{i}.pt"))
            grad_diff = torch.norm(grad_1 - grad_2, p="fro") / torch.norm(grad_1, p="fro")
            grad_diff_lst.append(grad_diff.item())
        row_mean.append(np.mean(grad_diff_lst))
        row_std.append(np.std(grad_diff_lst))

    data_mean.append(row_mean)
    data_std.append(row_std)

data_mean_df = pd.DataFrame(data_mean, index = grad_path_dict.keys(), columns = grad_path_dict.keys())
print("mean: ", data_mean_df)
data_std_df = pd.DataFrame(data_std, index = grad_path_dict.keys(), columns = grad_path_dict.keys())
print("std: ", data_std_df)
