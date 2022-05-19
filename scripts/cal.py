import pandas as pd

DATA_PATH = "/data2/zzp1012/batch-norm-exp4/outs/0515/exp4/LeNet5/0515-120926-seed0-mnist-LeNet5_v3-epochs10-lr0.01-bs128-wd0.0-momentum0.0/exp/0516-211447-seed0-mnist-LeNet5_v3-bs128/exp4/loss.csv"

data = pd.read_csv(DATA_PATH)
diff_linear = data["H_off_dot_Y_linear_norm"] / data["H_off_dot_Y_norm"]
print("linear mean: ", diff_linear.mean())
print("liear std: ", diff_linear.std())
diff_none = data["H_off_dot_Y_none_norm"] / data["H_off_dot_Y_norm"]
print("none mean: ", diff_none.mean())
print("none std: ", diff_none.std())
