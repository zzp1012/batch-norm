import os
import argparse
import torch
import torch.nn as nn
from typing import NoReturn

# import internal libs
from utils import set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from config import DATE, MOMENT, SRC_PATH

# define the forward pass of the network
class Net(nn.Module):
    def __init__(self, 
                 d: int,
                 isBN: bool) -> NoReturn:
        """simple net, include a linear layer and BN(if isBN is True)

        Args:
            d (int): the dimension of input.
            isBN (bool): if use BN.
        
        Return:
            None
        """
        super(Net, self).__init__()

        # define the linear layer
        self.main = nn.Sequential(
            nn.Linear(d, 1), # (N, d) -> (N, 1)
            nn.BatchNorm1d(1, eps=0) if isBN else nn.Identity(), # (N, 1) -> (N, 1)
        )

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
    # loss = 1 / n * sum(1 + 2y + 3y^2 + 4y^3 + 5y^4)
    losses = 1 + 2 * y + 3 * torch.pow(y, 2) + 4 * torch.pow(y, 3) + 5 * torch.pow(y, 4)
    loss = torch.mean(losses)
    return loss


def train(device: torch.device,
          save_path: str,
          model: Net,
          inputs: torch.Tensor,
          epochs: float,
          lr: float) -> NoReturn:
    """
    Args:
        device (torch.device): the device to train the model.
        save_path (str): the path to save the model.
        model (Net): the model
        inputs (tensor): the input
        epochs (float): number of epochs
        lr (float): learning rate
    
    Return:
        None
    """
    logger = get_logger("train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # put model and input on device
    model = model.to(device)
    inputs = inputs.to(device)
    # set the model to train mode
    model.train()
    # define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    # define the number of epochs
    for epoch in range(epochs):
        # forward pass
        y = model(inputs)
        # compute the loss
        loss = loss_fn(y)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print the loss
        logger.info("epoch: {}, loss: {}".format(epoch, loss.item()))


# generate random tensor, called Z of shape (n, d)
def generate_Z(n: int, 
               d: int, 
               low: int = -10, 
               high: int = 10):
    """generate random tensor, called Z of shape (n, d)"""
    logger = get_logger("generate_Z")
    Z = torch.randint(low, high, size = (n, d)).float()
    logger.info(f"Z shape: {Z.shape}; max: {Z.max()}; min: {Z.min()}")
    return Z


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
    parser.add_argument("-n", '--sample_num', default=100, type=int,
                        help="set the number of inputs.")
    parser.add_argument("-d", '--input_dim', default=1, type=int,
                        help="set the dimension of inputs.")
    parser.add_argument("-e", '--epochs', default=10, type=int,
                        help="set epoch number")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="set the learning rate.")
    parser.add_argument("-b", "--is_bn", action="store_true", dest="is_bn",
                        help="enable BN.")
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
                         "bn" if args.is_bn else "no_bn",
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
    logger.info("#########generating inputs....")
    inputs = generate_Z(n = args.sample_num, d = args.input_dim)
    
    # define the model
    logger.info("#########define the model....")
    model = Net(d = args.input_dim, isBN = args.is_bn)
    
    # train the model
    logger.info("#########training the model....")
    train(device = args.device,
          save_path = os.path.join(args.save_path, "exp"),
          model = model,
          inputs = inputs,
          epochs = args.epochs,
          lr = args.lr)

if __name__ == "__main__":
    main()