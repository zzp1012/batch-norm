import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd.functional import hessian
from torch.utils.data import Dataset, DataLoader
from typing import NoReturn
from tqdm import tqdm

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from config import DATE, MOMENT, SRC_PATH


def create_batches(dataset: Dataset,
                   batch_size: int,
                   seed: int) -> list:
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
    batch_indices = []
    repeat_num = 100
    for itr in range(1, repeat_num+1):
        for i, label in enumerate(range(len(dataset.classes))):
            indices = np.where(labels == label)[0]
            random.Random(seed + itr + i).shuffle(indices)
            batch_indices.append(indices[:batch_size])
    # create the batches
    batches = []
    for idx in batch_indices:
        batches.append((inputs[idx], labels[idx]))
    return batches


def test(save_path: str,
         device: torch.device,
         model: nn.Module,
         testset: Dataset,
         hessian_mat_dict: dict,
         batch_size: int,
         seed: int) -> NoReturn:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to test
        testset: the test dataset
        hessian_mat_dict: the hessian matrix
        pos_lbl: the label of positive samples.
        neg_lbl: the label of negative samples.
        batch_size: the batch size
        seed: the seed
    """
    logger = get_logger(__name__)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # put the model to GPU or CPU
    model = model.to(device)
    # create test batches
    test_batches = create_batches(testset, batch_size, seed)
    
    ## add the hook things
    # single forward one
    features = []
    def hook(module, input, output):
        assert len(input) == 1, "the input should be a tuple containing one tensor"
        features.append(input[0])
    forward_handle = model.bn.register_forward_hook(hook)
    
    # initialize the final res dict
    loss_dict = {
        "L_d_linear": [],
        "L_d_none": [],
        "Y_linear_dot_y_d_norm": [],
        "Y_none_dot_y_d_norm": [],
        "Y_linear_norm": [],
        "Y_none_norm": [],
        "H_off_dot_Y_linear_norm": [],
        "H_off_dot_Y_none_norm": [],
    }  
    
    # get the two loss terms
    for batch_idx, (inputs, labels) in enumerate(test_batches):
        logger.info(f"#####batch {batch_idx}")
        # set the inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        assert len(labels.unique()) == 1, "the labels should be the same"
        label = labels.unique().item()
        logger.info(f"label: {label}")
        
        model.eval()
        with torch.no_grad():
            # set the outputs
            outputs = model(inputs) # (N, 1)
            print((outputs.max(1)[1] == labels).float().mean().item())
            
            # get the features
            assert len(features) == 1, \
                "the features should be a tuple containing one tensor"
            feature = features[-1]
            del features[-1]
            
            X = feature.T
            D, N = X.shape

            # get the Y
            batch_mean = torch.mean(X, dim=-1, keepdim=True) # (D, 1)
            batch_var = torch.var(X, dim=-1, unbiased=False, keepdim=True) # (D, 1)
            Y = (X - batch_mean) / torch.sqrt(batch_var + 1e-5) # (D, N)

            # remove the rows that are all 0
            none_zero_rows = torch.where(torch.sum(Y**2, dim=-1) != 0)[0]
            # record the list
            L_d_linear_lst, L_d_none_lst, Y_linear_dot_y_d_norm_lst, Y_none_dot_y_d_norm_lst, \
                Y_linear_norm_lst, Y_none_norm_lst, H_off_dot_Y_linear_norm_lst, H_off_dot_Y_none_norm_lst = [], [], [], [], [], [], [], []
            for d in tqdm(none_zero_rows):
                # calculate quantities
                y_d = Y[d, :] # (N, )
                H_off_d = hessian_mat_dict[label].fill_diagonal_(0)[d, :].to(device) # (D, )
                A = torch.diag(torch.norm(Y, dim = -1, p = 2)) # (D, D)
                lambda_d = torch.cosine_similarity(Y, y_d.repeat(D, 1)) # (D, )
                
                # get Y_linear and Y_none
                Y_linear = torch.matmul(torch.matmul(A, lambda_d.reshape(D, 1)), y_d.reshape(1, N) / torch.norm(y_d, p=2)) # (D, N)
                Y_none = Y - Y_linear
                Y_none = Y_none - torch.mean(Y_none, dim=-1).unsqueeze(-1) # (D, )

                # get the partial loss
                Y_linear_dot_y_d = torch.matmul(Y_linear, y_d.reshape(N, 1)) # (D, 1)
                Y_none_dot_y_d = torch.matmul(Y_none, y_d.reshape(N, 1)) # (D, 1)

                # get the gradient actually
                H_off_dot_Y_linear = torch.matmul(H_off_d.reshape(1, D), Y_linear) # (1, D)
                H_off_dot_Y_none = torch.matmul(H_off_d.reshape(1, D), Y_none) # (1, D)
                
                # calculate the two loss
                L_d_linear = torch.matmul(H_off_d.reshape(1, D), Y_linear_dot_y_d) # (1, 1)
                L_d_none = torch.matmul(H_off_d.reshape(1, D), Y_none_dot_y_d) # (1, 1)

                # record the quantities
                L_d_linear_lst.append(L_d_linear.item())
                L_d_none_lst.append(L_d_none.item())
                Y_linear_dot_y_d_norm_lst.append(torch.norm(Y_linear_dot_y_d, p=2).item())
                Y_none_dot_y_d_norm_lst.append(torch.norm(Y_none_dot_y_d, p=2).item())
                Y_linear_norm_lst.append(torch.norm(Y_linear, p="fro").item())
                Y_none_norm_lst.append(torch.norm(Y_none, p="fro").item())
                H_off_dot_Y_linear_norm_lst.append(torch.norm(H_off_dot_Y_linear, p="fro").item())
                H_off_dot_Y_none_norm_lst.append(torch.norm(H_off_dot_Y_none, p="fro").item())

            # save the results
            loss_dict["L_d_linear"].append(np.mean(L_d_linear_lst))
            loss_dict["L_d_none"].append(np.mean(L_d_none_lst))
            loss_dict["Y_linear_dot_y_d_norm"].append(np.mean(Y_linear_dot_y_d_norm_lst))
            loss_dict["Y_none_dot_y_d_norm"].append(np.mean(Y_none_dot_y_d_norm_lst))
            loss_dict["Y_linear_norm"].append(np.mean(Y_linear_norm_lst))
            loss_dict["Y_none_norm"].append(np.mean(Y_none_norm_lst))
            loss_dict["H_off_dot_Y_linear_norm"].append(np.mean(H_off_dot_Y_linear_norm_lst))
            loss_dict["H_off_dot_Y_none_norm"].append(np.mean(H_off_dot_Y_none_norm_lst))
    
    # save the gradients
    loss_df = pd.DataFrame.from_dict(loss_dict)
    loss_df.to_csv(os.path.join(save_path, "loss.csv"), index = False)
    
    # remove the hook
    forward_handle.remove()


def get_hessian(model: nn.Module,
                device: torch.device,
                dataset: torch.utils.data.Dataset,):
    """the model could calculate the hessian matrix.

    Args:
        model: the model to calculate the hessian matrix.
        device: GPU or CPU
    
    Return:
        Hessian
    """
    model = model.to(device)
    model.eval()

    # get the input shape of after_bn
    _, D = model.after_bn[0].weight.shape
    # make the input
    x = torch.zeros(1, D).to(device)
    
    # calculate the hessian
    hessian_mat_dict = dict()
    for lbl in range(len(dataset.classes)):
        def model_with_loss(x):
            """the model with loss.
            """
            y_hat = model.after_bn(x)
            y = torch.tensor([lbl]).to(device)
            loss = nn.CrossEntropyLoss(reduction="none")(y_hat, y)
            return loss
        
        # calculate the hessian
        hessian_mat = hessian(model_with_loss, x)
        # save the hessian
        hessian_mat_dict[lbl] = hessian_mat.squeeze().clone().detach()
    return hessian_mat_dict


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="hessian calculation")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the random seed.")
    parser.add_argument("--resume_path", default=None, type=str,
                        help='the path of pretrained model.')
    parser.add_argument("--dataset", default="mnist", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="AlexNet", type=str,
                        help='the model name.')
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    # set the save_path
    exp_name = "-".join([DATE, 
                         MOMENT,
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"bs{args.bs}"])
    args.save_path = os.path.join(os.path.dirname(args.resume_path), exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)

    # show the args.
    logger.info("#########parameters settings....")
    import config
    log_settings(args, config.__dict__)

    # save the current src
    save_current_src(save_path = args.save_path, 
                     src_path = SRC_PATH)

    # prepare the dataset
    logger.info("#########preparing dataset....")
    trainset, testset = prepare_dataset(args.dataset)

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset)
    logger.info(f"load the pretrained model from {args.resume_path}")
    model.load_state_dict(torch.load(args.resume_path))
    logger.info(model)

    # calculate the hessian
    logger.info("#########calculating hessian....")
    hessian_mat_dict = get_hessian(model, args.device, testset)

    # test
    logger.info("#########testing....")
    test(save_path = os.path.join(args.save_path, "exp4"),
         device = args.device,
         model = model,
         testset = testset,
         hessian_mat_dict = hessian_mat_dict,
         batch_size = args.bs,
         seed = args.seed)


if __name__ == "__main__":
    main()