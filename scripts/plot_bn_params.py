import os
import torch
import numpy as np
from matplotlib import pyplot as plt

def plot_multiple_curves(save_path: str, 
                         res_dict: dict,
                         name: str) -> None:
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
    for key in res_dict.keys():
        ax.plot(np.arange(len(res_dict[key])) + 1, np.asarray(res_dict[key]), label=key)
    ax.grid()
    ax.set(xlabel = 'epoch', title = name)
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(frameon=True, prop={'size': 10})
    # save the fig
    path = os.path.join(save_path, "{}.png".format(name))
    fig.savefig(path)
    plt.close()

res_path = "NEED TO BE FILLED"
plot_dim = 15
epochs = 100

for dir_name in os.listdir(res_path):
    for postfix in ["mean", "var"]:
        val_lst = []
        for i in range(1, epochs+1):
            val = torch.load(os.path.join(res_path, dir_name, f"{i}_{postfix}.pth"))
            val_lst.append(val)
        val_lst = torch.stack(val_lst).cpu().numpy()
        N, D = val_lst.shape
        assert N == epochs, "the number of epochs is not correct"

        val_dict = {f"dim{i}": val_lst[:, i] for i in range(min(D, plot_dim))}

        save_path = os.path.join(res_path, dir_name)
        os.makedirs(save_path, exist_ok=True)
        plot_multiple_curves(save_path = save_path,
                             res_dict = val_dict,
                             name = f"{postfix}_curve")