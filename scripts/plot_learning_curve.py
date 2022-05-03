import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_multiple_curves(save_path: str, 
                         res_dict: dict,
                         name: str,
                         ylim: list) -> None:
    """plot curves in one figure for each key in dictionary.
    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
        name (str): the name of the plot.
    """
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=(7, 5))
    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend.
    for key, res in res_dict.items():
        ax.plot(list(res.keys()), list(res.values()), label=key)
    ax.grid()
    ax.set(xlabel = 'epoch', title = name)
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(frameon=True, prop={'size': 10})
    plt.ylim(ylim)
    # save the fig
    path = os.path.join(save_path, "{}.png".format(name))
    fig.savefig(path)
    plt.close()

# load the data
data_path = "NEED TO BE FILLED"
data = pd.read_csv(data_path)

# make the res dict
loss_dict = {key: val for key, val in data.to_dict().items() if key in ["train_loss", "test_loss"]}
acc_dict = {key: val for key, val in data.to_dict().items() if key in ["train_acc", "test_acc"]}
cosine_dict = {key: val for key, val in data.to_dict().items() if key in ["cosine"]}

# make save path
save_path = os.path.dirname(data_path)

# plot
plot_multiple_curves(save_path = save_path,
                     res_dict = loss_dict,
                     name = "loss-curve",
                     ylim = [0.6, 2.6])

plot_multiple_curves(save_path = save_path,
                     res_dict = acc_dict,
                     name = "acc-curve",
                     ylim = [0, 1])

plot_multiple_curves(save_path = save_path,
                     res_dict = cosine_dict,
                     name = "cosine-curve",
                     ylim = [0, 1])