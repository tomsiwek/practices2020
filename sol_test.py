#!/home/ts417377/miniconda3/envs/nec_sol/bin/python 
import torch                                      # load PyTorch
from torchvision import models                    # load PyTorch model zoo
import sol.pytorch as sol                         # load SOL library

sol.config.print()

py_model  = models.__dict__['densenet201']()      # initialize DenseNet 201
input     = torch.rand(32, 3, 224, 224)           # initialize random input data
sol_model = sol.optimize(py_model, input.size())  # optimize py_model using SOL
sol_model.load_state_dict(py_model.state_dict())  # load parameters of py_model into sol_model
sol.device.set(sol.device.ve, 0)
output    = sol_model(input)                      # run sol_model
