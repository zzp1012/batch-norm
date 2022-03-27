import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import NoReturn

# import internal libs
from utils import set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src, update_dict
from utils.custom_bn import BatchNorm1d
from config import DATE, MOMENT, SRC_PATH, LOSS_LOWER_BOUND

# define the forward pass of the network
class Net(nn.Module):
    def __init__(self, 
                 d: int,
                 isCustomBN: bool) -> NoReturn:
        """simple net, include a linear layer and BN(if isBN is True)

        Args:
            d (int): the dimension of input.
            isCustomBN (bool): if use torch BN.
        
        Return:
            None
        """
        super(Net, self).__init__()

        # define the linear layer
        self.main = nn.Sequential(
            nn.Linear(d, 1), # (N, d) -> (N, 1)
            BatchNorm1d(1, eps = 0) if isCustomBN else nn.BatchNorm1d(1, eps = 0), # (N, 1) -> (N, 1)
        )

        # initialize the weights...
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, 
                z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (tensor): input (N, d)
        
        Return:
            x (tensor): output (N, 1)
        """
        return self.main(z)


# define the loss function
def loss_fn(y: torch.Tensor) -> torch.Tensor:
    """loss function

    Args:
        y (tensor): of shape (n, 1)
    
    Return:
        loss (tensor): scalar
    """
    losses =  - 3 * torch.pow(y, 2) # (n, 1)
    return losses


def train(device: torch.device,
          save_path: str,
          model: Net,
          Z: torch.Tensor,
          epochs: float,
          lr: float,
          bs: int,
          isLearn: bool,
          lower_bound: float = LOSS_LOWER_BOUND) -> Net:
    """
    Args:
        device (torch.device): GPU
        save_path (str): the path to save the model.
        model (Net): the model, already on device.
        Z (tensor): the input
        epochs (float): number of epochs
        lr (float): learning rate
        bs (int): batch size
        isLearn (bool): if the learning the learnable parameters in BN.
        lower_bound (float): the lower bound of the loss.
    
    Return:
        model (Net): the trained model.
    """
    logger = get_logger("train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # assertion
    assert len(Z.shape) == 2, "the shape of Z should be (n, d)"
    N, D = Z.shape

    # define if the model's BN's learnable parameters is learnable
    if not isLearn:
        model.main[1].weight.requires_grad = False
        model.main[1].bias.requires_grad = False
    else:
        model.main[1].weight.requires_grad = True
        model.main[1].bias.requires_grad = True
    
    # define the optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

    # first forward
    Z_all = Z.to(device)
    _ = model(Z_all)
    assert model.main[1].num_batches_tracked == 1, \
        "check if the num_batches_tracked correctly updated."
    del Z_all

    # initialize the res_dict
    total_res_dict = {}
    # set the model to train mode
    model.train()
    # define the number of epochs
    for epoch in range(epochs):
        # shuffle
        np.random.seed(epoch) 
        indexes = np.arange(N)
        np.random.shuffle(indexes)
        batches = np.array_split(indexes, round(N / bs))

        # define the loss
        loss_lst = []
        for batch in batches:
            # to device
            z_bs = Z[batch].to(device)
            # forward
            y_bs = model(z_bs)
            # loss
            losses = loss_fn(y_bs)
            loss = torch.mean(losses)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the loss
            loss_lst.extend(losses.detach().cpu().numpy().reshape(-1))

        # calculate the total loss
        total_loss = np.mean(loss_lst)
        # print the loss
        logger.info("epoch: {}, loss: {}".format(epoch, total_loss))

        # update the res_dict
        res_dict = {
            "epoch": [epoch],
            "loss": [total_loss],
        }
        total_res_dict = update_dict(src = res_dict,
                                     dst = total_res_dict)

        # save the model
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{epoch}.pth"))

        # check if go on
        if total_loss < lower_bound:
            logger.warning(f"the total loss {total_loss} has reached the lower bound {lower_bound}. Break training loop.")
            break
   
    # save the inputs
    torch.save(Z, os.path.join(save_path, "Z.pt"))
    # save the res_dict
    res_df = pd.DataFrame(total_res_dict)
    res_df.to_csv(os.path.join(save_path, "train.csv"), index = False)

    return model


def test(device: torch.device,
         save_path: str,
         model: Net,
         Z: torch.Tensor,
         bs: int) -> NoReturn:
    """test the trained model

    Args:
        device (torch.device): GPU
        save_path (str): the path to save the model.
        model (Net): the model, already on device.
        Z (tensor): the input
        bs (int): batch size
    
    Return:
        None
    """
    logger = get_logger("test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # assertion
    assert len(Z.shape) == 2, "the shape of Z should be (n, d)"
    N, D = Z.shape

    # set the model to eval mode
    model.eval()
    # define the number of batches
    batches = np.array_split(np.arange(N), round(N / bs))
    # define the loss
    loss_lst = []
    for batch in batches:
        # to device
        z_bs = Z[batch].to(device)
        # forward
        y_bs = model(z_bs)
        # loss
        losses = loss_fn(y_bs)

        # record the loss
        loss_lst.extend(losses.detach().cpu().numpy().reshape(-1))
    
    # calculate the total loss
    total_loss = np.mean(loss_lst)
    # print the loss
    logger.info("loss: {}".format(total_loss))

    # store the loss_lst
    np.save(os.path.join(save_path, "loss_lst.npy"), np.array(loss_lst))
    # save the input
    torch.save(Z, os.path.join(save_path, "Z.pt"))


# generate random tensor, called Z of shape (n, d)
def generate_Z(n: int, 
               d: int, 
               seed: int = 0,
               low: int = -10, 
               high: int = 10):
    """generate random tensor, called Z of shape (n, d)"""
    logger = get_logger("generate_Z")
    set_seed(seed) # set the seed
    Z = torch.rand(size = (n, d)) * (high - low) + low
    logger.info(f"Z shape: {Z.shape}; max: {Z.max()}; min: {Z.min()}")
    return Z.float()


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the random seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("-n", '--sample_num', default=1000, type=int,
                        help="set the number of inputs.")
    parser.add_argument("-d", '--input_dim', default=10, type=int,
                        help="set the dimension of inputs.")
    parser.add_argument("-e", '--epochs', default=10, type=int,
                        help="set epoch number")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="set the learning rate.")
    parser.add_argument("--bs", default=100, type=int,
                        help="set the batch size")
    parser.add_argument("-b", "--is_custom_bn", action="store_true", dest="is_custom_bn",
                        help="enable custom BN.")
    parser.add_argument("-t", "--is_learn", action="store_true", dest="is_learn",
                        help="enable training the learnable parameters in BN.")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([DATE, 
                         MOMENT,
                         f"seed{args.seed}",
                         "custom_bn" if args.is_custom_bn else "torch_bn",
                         "learnable" if args.is_learn else "not_learnable",
                         f"sample_num{args.sample_num}",
                         f"input_dim{args.input_dim}",
                         f"epochs{args.epochs}",
                         f"lr{args.lr}",])
    args.save_path = os.path.join(args.save_root, exp_name)
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

    # generate the inputs
    logger.info("#########generating train set....")
    Z_train = generate_Z(n = args.sample_num, 
                         d = args.input_dim,
                         seed = args.seed) # the data on cpu now.
    
    logger.info("#########generating test set....")
    Z_test = generate_Z(n = args.sample_num,
                        d = args.input_dim,
                        seed = (args.seed+1) ** 2,
                        low = -20,
                        high = 20)
    
    # define the model
    logger.info("#########define the model....")
    model = Net(d = args.input_dim, isCustomBN = args.is_custom_bn)
    model = model.to(args.device)
    
    # train the model
    logger.info("#########training the model....")
    model = train(device = args.device,
                  save_path = os.path.join(args.save_path, "train"),
                  model = model,
                  Z = Z_train,
                  epochs = args.epochs,
                  lr = args.lr,
                  bs = args.bs,
                  isLearn = args.is_learn)

    # test the model
    logger.info("#########testing the model....")
    test(device = args.device,
         save_path = os.path.join(args.save_path, "test"),
         model = model,
         Z = Z_test,
         bs = args.bs)


if __name__ == "__main__":
    main()