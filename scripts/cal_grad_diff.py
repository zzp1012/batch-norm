import torch
import pandas as pd

# set the global path
loss1_grad_path = "/lustre/home/acct-eezqs/eezqs/zzp1012/batch-norm-exp2/outs/0511/loss1/0511-134717-seed0-cifar10-vgg11-loss1-bs128/exp2/classifier.0_grads.pt"
loss2_grad_path = "/lustre/home/acct-eezqs/eezqs/zzp1012/batch-norm-exp2/outs/0511/loss2/0511-140218-seed0-cifar10-vgg11-loss2-bs128/exp2/classifier.0_grads.pt"
loss3_grad_path = "/lustre/home/acct-eezqs/eezqs/zzp1012/batch-norm-exp2/outs/0511/loss3/0511-134849-seed0-cifar10-vgg11-loss3-bs128/exp2/classifier.0_grads.pt"
loss4_grad_path = "/lustre/home/acct-eezqs/eezqs/zzp1012/batch-norm-exp2/outs/0511/loss4/0511-134935-seed0-cifar10-vgg11-loss4-bs128/exp2/classifier.0_grads.pt"

# load the grad
loss1_grad = torch.load(loss1_grad_path)
print(loss1_grad.shape)
loss2_grad = torch.load(loss2_grad_path)
print(loss1_grad.shape)
loss3_grad = torch.load(loss3_grad_path)
print(loss1_grad.shape)
loss4_grad = torch.load(loss4_grad_path)
print(loss1_grad.shape)
loss_grad_dict = {
    "loss1": loss1_grad,
    "loss2": loss2_grad,
    "loss3": loss3_grad,
    "loss4": loss4_grad,
}

# create a table
data = []
for loss_type_1, grad_1 in loss_grad_dict.items():
    print(f"row: {loss_type_1}")
    row = []
    for loss_type_2, grad_2 in loss_grad_dict.items():
        print(f"col: {loss_type_2}")
        assert len(grad_2.shape) == 2, "the grad should be 2-dim"
        grad_diff_norm = torch.norm(grad_1 - grad_2, p=2, dim=-1) / torch.norm(grad_1, p=2, dim=-1) # (N, )
        row.append(torch.mean(grad_diff_norm).item())
    data.append(row)

data_df = pd.DataFrame(data, index = loss_grad_dict.keys(), columns = loss_grad_dict.keys())
print(data_df)