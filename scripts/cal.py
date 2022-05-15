import pandas as pd

DATA_PATH = "NEED TO BE FILLED"

data = pd.read_csv(DATA_PATH)
diff = data["H_off_dot_Y_linear_norm"] - data["H_off_dot_Y_none_norm"]
print("mean: ", diff.mean())
print("std: ", diff.std())
print(data.describe())
