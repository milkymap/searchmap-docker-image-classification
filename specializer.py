import torch as th 
import torch.nn as nn 

import operator as op 
import functools as ft, itertools as it

class NRLModel(nn.Module):
    def __init__(self):
        super(NRLModel, self).__init__()
        self.activation_map = {
            0: nn.Identity(), 
            1: nn.ReLU(), 
            2: nn.LeakyReLU(0.2),
            3: nn.GELU(), 
            4: nn.Tanh(), 
            5: nn.Sigmoid(), 
            6: nn.Softmax(dim=-1)
        }

class MLPModel(NRLModel):
    def __init__(self, layers_config, activations, apply_norm=False):
        super(MLPModel, self).__init__()

        self.shapes = list(zip(layers_config[:-1], layers_config[1:]))
        self.activations = activations 
        self.linear_layers = nn.ModuleList([]) 
        
        for index, (in_dim, out_dim) in enumerate(self.shapes):
            linear = nn.Linear(in_features=in_dim, out_features=out_dim)
            nonlinear = self.activation_map[self.activations[index]] 
            if apply_norm and index < len(self.shapes) - 1:
                normalizer = nn.BatchNorm1d(out_dim)
                layer = nn.Sequential(linear, normalizer, nonlinear)
            else:
                layer = nn.Sequential(linear, nonlinear)
            self.linear_layers.append(layer)
    
    def forward(self, X_0):
        X_N = ft.reduce(lambda X_I, LIN: LIN(X_I), self.linear_layers, X_0)
        return X_N
