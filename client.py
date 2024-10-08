from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from utils import * 

from hydra.utils import instantiate

import torch
import flwr as fl

from model import train, test
from vmunet_v2 import VMUNetV2
from train import training_process
from test import testing_process
from test_v2 import test_v2

#from flwr.common import Context



class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient."""

    def __init__(self, trainloader, valloader, model_cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        # For further flexibility, we don't hardcode the type of model we use in
        # federation. Here we are instantiating the object defined in `conf/model/net.yaml`
        # (unless you changed the default) and by then `num_classes` would already be auto-resolved
        # to `num_classes=10` (since this was known right from the moment you launched the experiment)
        
        
        # defining model 
        print(1111111)
        self.model = VMUNetV2(deep_supervision= model_cfg.deep_supervision,
                              depths=model_cfg.depths,
                              depths_decoder=model_cfg.depths_decoder,
                              drop_path_rate=model_cfg.drop_path_rate,
                              input_channels=model_cfg.input_channels,
                              load_ckpt_path=model_cfg.load_ckpt_path,
                              num_classes=model_cfg.num_classes)
        # device = 'cpu:0'
        # print(torch.cuda.is_available())
        # device = 'cuda:0'

        self.model = self.model.cuda()
        self.model.load_from()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        it1 = self.model.state_dict().items()
        it2 = state_dict.items()
        l1 = len(it1)
        l2 = len(it2)
        if l1!=l2 :
            print(f"{l1} : {l2} length do not match")
        else :
            for i in self.model.state_dict() :
                if not self.model.state_dict()[i].shape == state_dict[i].shape:
                    print(i, self.model.state_dict()[i].shape, state_dict[i].shape,"Different")
        print(2222222)


        # state_dict = OrderedDict()

        # for key, param in params_dict:
        #     param_tensor = torch.tensor(param, dtype=torch.float32)

        # # # Debugging print to check shapes
        # # print(f"Setting Parameter: {key}, Original Shape: {param_tensor.shape}")

        # # Handle potential shape mismatches
        # if param_tensor.shape == torch.Size([0]):
        #     param_tensor = param_tensor.reshape(())
        #     # print(f"Reshaped Parameter: {key}, New Shape: {param_tensor.shape}")

        # state_dict[key] = param_tensor

        # # Print shapes after state_dict is created
        # # print("##### Set Parameter Shapes #####")
        # # for name, tensor in state_dict.items():
        # #     print(f"Set Parameter: {name}, Shape: {tensor.shape}")




        self.model.load_state_dict(state_dict, strict=False)






    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
      parameters = [val.cpu().numpy() for val in self.model.state_dict().values()]
      # for param in parameters:
      #     print(f"Parameter shape: {param.shape}")
      return parameters



    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        print(121212121212)
        # config fit params from server
        lr = config["lr"]
        local_epochs = config["local_epochs"]
        weight_decay = config["weight_decay"]
        amsgrad = config["amsgrad"]
        betas = config["betas"]
        eps = config["eps"]
        T_max = config["T_max"]
        eta_min = config["eta_min"]
        last_epoch = config["last_epoch"]

        print(333333)

        # criterion
        criterion = BceDiceLoss(wb=1, wd=1)

        # optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                  lr=lr, 
                                  amsgrad=amsgrad, 
                                  betas=betas, 
                                  eps=eps, 
                                  weight_decay=weight_decay, 
                                  )

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = T_max,
            eta_min = eta_min,
            last_epoch = last_epoch
        )




        # do local training
        #train(self.model, self.trainloader, optim, epochs, self.device)

        training_process(train_loader= self.trainloader,
                         model= self.model,
                         criterion= criterion, 
                         optimizer= optimizer,
                         scheduler= scheduler,
                         local_epochs= local_epochs, 
                         step= 0)
        print(44444444)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        criterion = BceDiceLoss(wb=1, wd=1)

        loss , metrics = testing_process(val_loader=self.valloader,
                        model= self.model,
                        criterion= criterion
                        )

        # loss, metrics = test_v2(self.model, self.valloader, self.device, criterion)
        

        return float(loss), len(self.valloader), {"accuracy": metrics[0],
                                                  "sensitivity": metrics[1],
                                                  "specificity": metrics[2],
                                                  "f1_or_dsc": metrics[3],
                                                  "miou": metrics[4]}

    print(5555555)


def generate_client_fn(trainloaders, valloaders, model_cfg):
    """Return a function to construct a FlowerClient."""

    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            model_cfg=model_cfg,
        )#.to_client()

    return client_fn