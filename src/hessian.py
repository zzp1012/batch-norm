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
                   seed: int,
                   pos_lbl: int,
                   neg_lbl: int,) -> list:
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
    repeat_num = 300
    for itr in range(1, repeat_num+1):
        for i, label in enumerate(range(len(dataset.classes))):
            if label == pos_lbl or label == neg_lbl:    
                indices = np.where(labels == label)[0]
                random.Random(seed + itr + i).shuffle(indices)
                batch_indices.append(indices[:batch_size])
            else:
                continue
    # make pos lablels to be 1
    labels[labels == pos_lbl] = 1
    # make neg lablels to be 0
    labels[labels == neg_lbl] = 0
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
         pos_lbl: int,
         neg_lbl: int,
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
    test_batches = create_batches(testset, batch_size, seed, pos_lbl, neg_lbl)
    
    ## add the hook things
    # single forward one
    features = []
    def hook(module, input, output):
        assert len(input) == 1, "the input should be a tuple containing one tensor"
        features.append(input[0])
    forward_handle = model.after_bn.register_forward_hook(hook)
    # backward
    layers = {
        "x": model.before_bn,
        "y": model.bn,
    }
    grads = {}
    backward_handles = {}
    for layer_name, module in layers.items():
        def register_hook(module, layer_name):
            def get_grads(module, grad_input, grad_output):
                assert len(grad_output) == 1, f"the grad output length should be 1, but got {len(grad_output)}"
                grads[layer_name] = grad_output[0].clone().detach()
            backward_handles[layer_name] = module.register_backward_hook(get_grads)
        register_hook(module, layer_name)
    
    # initialize the final res dict
    grad_norm_dict = {
        "loss_linear_y_grad": [],
        "loss_linear_x_grad": [],
        "loss_none_y_grad": [],
        "loss_none_x_grad": []
    }
    # train the model
    model.train()
    for batch_idx, (inputs, labels) in enumerate(tqdm(test_batches)):
        # set the inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        assert len(labels.unique()) == 1, "the labels should be the same"
        label = labels.unique().item()
        
        for loss_id in [1, 2]:
            # set the outputs
            outputs = model(inputs) # (N, 1)
            
            # get the features
            assert len(features) == 1, \
                "the features should be a tuple containing one tensor"
            feature = features[-1]
            del features[-1]
            
            # calculate loss
            Y = feature.T
            D, N = Y.shape
            y_0 = Y[0, :] # (N, )
            H_off_0 = hessian_mat_dict[label].fill_diagonal_(0)[0, :].to(device).detach() # (D, )
            A = torch.diag(torch.norm(Y, dim = -1, p = 2)).detach() # (D, D)
            lambda_0 = torch.cosine_similarity(Y, y_0.repeat(D, 1)).detach() # (D, )
            Y_linear = torch.matmul(torch.matmul(A, lambda_0.reshape(D, 1)), y_0.reshape(1, N) / torch.norm(y_0, p=2)) # (D, N)
            if loss_id == 1: 
                loss = torch.matmul(torch.matmul(H_off_0.reshape(1, D), Y_linear), y_0.reshape(N, 1)) # (1, )
            elif loss_id == 2:
                loss = torch.matmul(torch.matmul(H_off_0.reshape(1, D), Y - Y_linear), y_0.reshape(N, 1)) # (1, )
            else:
                raise ValueError(f"the loss id should be 1 or 2, but got {loss_id}")
            
            # set the gradients to zero
            model.zero_grad()
            # backward
            loss.backward()
            
            # check and record the grad
            for layer_name, grad in grads.items():
                if layer_name == "y":
                    if loss_id == 1:
                        assert torch.allclose(torch.matmul(H_off_0.reshape(1, D), Y_linear).reshape(N), grad[:, 0])
                        grad_norm_dict["loss_linear_y_grad"].append(torch.norm(grad[:, 0]).item())
                    elif loss_id == 2:
                        assert torch.allclose(torch.matmul(H_off_0.reshape(1, D),Y -  Y_linear).reshape(N), grad[:, 0])
                        grad_norm_dict["loss_none_y_grad"].append(torch.norm(grad[:, 0]).item())
                    else:
                        raise ValueError(f"the loss id should be 1 or 2, but got {loss_id}")
                elif layer_name == "x":
                    if loss_id == 1:
                        grad_norm_dict["loss_linear_x_grad"].append(torch.norm(grad[:, 0]).item())
                    elif loss_id == 2:
                        grad_norm_dict["loss_none_x_grad"].append(torch.norm(grad[:, 0]).item())
                    else:
                        raise ValueError(f"the loss id should be 1 or 2, but got {loss_id}")
                else:
                    raise ValueError(f"the layer name should be x or y, but got {layer_name}")
    
    # save the gradients
    grad_norm_df = pd.DataFrame.from_dict(grad_norm_dict)
    grad_norm_df.to_csv(os.path.join(save_path, "grad_norm.csv"), index = False)
    
    # remove the hook
    forward_handle.remove()
    for _, handle in backward_handles.items():
        handle.remove()


def get_hessian(model: nn.Module,
                device: torch.device):
    """the model could calculate the hessian matrix.

    Args:
        model: the model to calculate the hessian matrix.
        device: GPU or CPU
    
    Return:
        Hessian
    """
    model = model.to(device)
    model.eval()
    hessian_mat_dict = dict()
    for lbl in [0., 1.]:
        def model_with_loss(x):
            """the model with loss.
            """
            y_hat = model.after_bn(x)
            y_hat = y_hat.reshape(y_hat.shape[0])
            y = torch.tensor([lbl]).to(device)
            loss = nn.BCEWithLogitsLoss(reduction="none")(y_hat, y)
            return loss
        
        # make the input
        x = torch.zeros(1024).to(device)
        # calculate the hessian
        hessian_mat = hessian(model_with_loss, x)
        # save the hessian
        hessian_mat_dict[lbl] = hessian_mat.clone().detach()
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
    parser.add_argument("--pos", default=1, type=int,
                        help="the postive label in two-cat classificiation problem")
    parser.add_argument("--neg", default=0, type=int,
                        help="the postive label in two-cat classificiation problem")
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
                         f"{args.pos}And{args.neg}",
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
    hessian_mat_dict = get_hessian(model, args.device)

    # test
    logger.info("#########testing....")
    test(save_path = os.path.join(args.save_path, "exp4"),
         device = args.device,
         model = model,
         testset = testset,
         hessian_mat_dict = hessian_mat_dict,
         pos_lbl = args.pos,
         neg_lbl = args.neg,
         batch_size = args.bs,
         seed = args.seed)


if __name__ == "__main__":
    main()