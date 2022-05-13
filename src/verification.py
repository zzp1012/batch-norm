import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import NoReturn
from tqdm import tqdm

# import internal libs
from utils import set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from config import DATE, MOMENT, SRC_PATH

# define the forward pass of the network
class Net(nn.Module):
    def __init__(self, 
                 d: int,) -> NoReturn:
        """simple net, include a linear layer and BN

        Args:
            d (int): the dimension of input.
        
        Return:
            None
        """
        super(Net, self).__init__()

        # define the linear layer
        self.main = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1), # (N, d) -> (N, 1)
            nn.BatchNorm1d(1,eps=0, affine=False) # (N, 1) -> (N, 1)
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
def loss_fn(y: torch.Tensor,
            loss_id: int,
            max_order: int,
            params: torch.Tensor) -> torch.Tensor:
    """loss function

    Args:
        y (tensor): of shape (n, 1)
        loss_id(int): the id of loss function
        max_order(int): the max order of the loss function
        params(tensor): the lamda of the ploy loss function
    
    Return:
        loss (tensor): scalar
    """
    assert len(params) == max_order+1,'does not match the shape'
    assert loss_id <= max_order + 1 and loss_id >= 1,'wrong loss id'
    losses = params[max_order] * torch.pow(y, max_order)
    for i in range(max_order-1,loss_id-2,-1):
        losses += params[i] * torch.pow(y,i)
    loss = torch.sum(losses)
    return loss


def test(device: torch.device,
         save_path: str,
         model: Net,
         inputs: float,
         itrs: int,
         max_order: int)-> NoReturn:
    """calculate the x_grad for each loss function

    Args:
        device: torch.device, the GPU to be used
        save_path: the save path for results
        model: Net
        inputs: float
        iter: the number of groups of loss functions
        max_order: the maximum order of the loss function

    Return:
        None
    """
    logger = get_logger("test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # put model and input on device
    model = model.to(device)
    inputs = inputs.to(device)
    
    # hook
    x_grads = []
    def get_grads(module, grad_input, grad_output):
        x_grads.append(grad_output[0].clone().detach().cpu().numpy())
    hook = model.main[-2].register_backward_hook(get_grads)
    print(model.main[-2])

    # register the grad and lamdas
    grads_list = []
    params_list = []
    # set the model to train mode
    model.train()
    # begin sample lamda
    for itr in tqdm(range(itrs)):
        params = torch.randn(max_order + 1)
        params_list.append(params.clone().detach().cpu().numpy())
        grads_piece = []
        # begin cal different loss id
        for loss_id in range(1, max_order + 2):
            # forward
            model.zero_grad()
            y = model(inputs)
            loss = loss_fn(y, loss_id, max_order, params)
            # backward
            loss.backward()
            # register a piece
            grads_piece.append(x_grads[-1])
            assert len(x_grads) == 1,'wrong grads len'
            del x_grads[0]
        # regiter a set
        grads_list.append(np.array(grads_piece))
    # remove the hook
    hook.remove()
    # save the data
    torch.save(inputs.detach(), os.path.join(save_path, "inputs.pt"))
    torch.save(model.main[0].weight.detach(),os.path.join(save_path, "weights.pt"))
    np.save(os.path.join(save_path, f"x_grads.npy"), np.array(grads_list))
    np.save(os.path.join(save_path, f"params.npy"), np.array(params_list))


# generate random tensor, called Z of shape (n, d)
def generate_Z(n: int, 
               d: int,):
    """generate random tensor, called Z of shape (n, d)"""
    logger = get_logger("generate_Z")
    Z = torch.randn(n, d, dtype=torch.float32)
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
    parser.add_argument("-d", '--input_dim', default=10, type=int,
                        help="set the dimension of inputs.")
    parser.add_argument("--itrs", default=1000, type=int,
                        help="set the sample iters")
    parser.add_argument("--max_order",default=4, type=int,
                        help="set the max order")
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
                         f"sample_num{args.sample_num}",
                         f"input_dim{args.input_dim}",
                         f"itrs{args.itrs}",
                         f"max_order{args.max_order}"])
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
    model = Net(d = args.input_dim)

    #test the model
    logger.info("#########training the model....")
    test(device = args.device,
          save_path = os.path.join(args.save_path, "exp"),
          model = model,
          inputs = inputs,
          itrs= args.itrs,
          max_order=args.max_order)

if __name__ == "__main__":
    main()