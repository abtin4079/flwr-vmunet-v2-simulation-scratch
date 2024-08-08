from collections import OrderedDict


from omegaconf import DictConfig
from hydra.utils import instantiate
from flwr.common import NDArrays, Scalar
from typing import Dict, Tuple

import torch

from model import Net, test
from vmunet_v2 import VMUNetV2
from test import testing_process
from utils import *
import os
from test_v2 import test_v2 




def get_on_fit_config(config: DictConfig):
    """Return a function to configure the client's fit."""

    def fit_config_fn(server_round: int):

        # change the learning rate with server_round
        # if server_round > 50:
        #     lr = config.lr / 10
        # else:
        #     lr = config.lr

        file_path = 'learning_rate.txt'
        if (os.path.exists(file_path) and server_round > 0):
            lr = read_lines_and_compute_mean(file_path)
        else:
            lr = config.lr

        return {
            "lr": lr,
            "betas"  : config.betas,
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "amsgrad" : config.amsgrad,
            "local_epochs": config.local_epochs,
            "T_max" : config.T_max,
            "eta_min" : config.eta_min,
            "last_epoch" : config.last_epoch
        }

    return fit_config_fn


def get_evalulate_fn(model_cfg: int, testloader):
    """Return a function to evaluate the global model."""

    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):

        # defining model 
        model = VMUNetV2(deep_supervision= model_cfg.deep_supervision,
                              depths=model_cfg.depths,
                              depths_decoder=model_cfg.depths_decoder,
                              drop_path_rate=model_cfg.drop_path_rate,
                              input_channels=model_cfg.input_channels,
                              load_ckpt_path=model_cfg.load_ckpt_path,
                              num_classes=model_cfg.num_classes)
                              
        # Print the shapes of the model's parameters just after initialization
        # print("##### Model Parameter Shapes #####")
        # for name, param in model.state_dict().items():
        #     print(f"Parameter: {name}, Shape: {param.shape}")



        device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
        model.to(device)
        
    # Convert parameters to model state_dict
        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        it1 = model.state_dict().items()
        it2 = state_dict.items()
        l1 = len(it1)
        l2 = len(it2)
        if l1!=l2 :
            print(f"{l1} : {l2} length do not match")
        else :
            for i in model.state_dict() :
                if not model.state_dict()[i].shape == state_dict[i].shape:
                    print(i, model.state_dict()[i].shape, state_dict[i].shape,"Different")
        


        #state_dict = OrderedDict()
        # for key, param in params_dict:
        #     param_tensor = torch.tensor(param, dtype=torch.float32)

        #     # Print shapes for debugging
        #     # print(f"Loading Parameter: {key}, Original Shape: {param_tensor.shape}")

        #     # Handle potential shape mismatches
        #     if param_tensor.shape == torch.Size([0]):
        #         # If the shape is [0], reshape it to []
        #         param_tensor = param_tensor.reshape(())
        #         # print(f"Reshaped Parameter: {key}, New Shape: {param_tensor.shape}")

        #     state_dict[key] = param_tensor

        # # Print shapes after state_dict is created
        # # print("##### Loaded Parameter Shapes #####")
        # # for name, tensor in state_dict.items():
        # #     print(f"Loaded Parameter: {name}, Shape: {tensor.shape}")



        model.load_state_dict(state_dict, strict=True)

        criterion = BceDiceLoss(wb=1, wd=1)


        loss , metrics = testing_process(val_loader=testloader,
                        model=model,
                        criterion= criterion
                        )   

        # loss, metrics = test_v2(model, testloader, device, criterion)     

        return float(loss), {"accuracy": metrics[0],
                             "sensitivity": metrics[1],
                             "specificity": metrics[2],
                             "f1_or_dsc": metrics[3],
                             "miou": metrics[4]}

    return evaluate_fn



def read_lines_and_compute_mean(file_path):
    with open(file_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()
    
    # Convert each line to a number
    numbers = [float(line.strip()) for line in lines]
    
    # Compute the mean
    mean = sum(numbers) / len(numbers)
    
    return mean