from xmlrpc.client import boolean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Iterable, Sized, Tuple

def truncated_normal_(tensor, mean: float = 0., std: float = 1.):  
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(out_features))

    def forward(self, x):
        raise NotImplementedError("abstract methodd called")

class ExpUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.uniform_(self.weight,a=-20.0, b=2.0)
        truncated_normal_(self.bias, std=0.5)
        self.size = in_features

    def forward(self, x):
        out = (x) @ torch.exp(self.weight) + self.bias
        return (1-0.01) * torch.clip(out, 0, 1) + 0.01 * out 
    
class ReLUUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        out = (x) @ self.weight + self.bias
        return F.relu(out)

class ConfluenceUnit(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)
        self.size = in_features

    def forward(self, x):
        out = (x) @ self.weight + self.bias
        return (1-0.01) * torch.clip(out, 0, 1) + 0.01 * out 

class FCLayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=-10.0, std=3)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        FC = x @ torch.exp(self.weight) + self.bias
        return FC

class FCLayer_notexp(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        FC = x @ self.weight + self.bias
        return FC

class ScalableMonotonicNeuralNetwork(torch.nn.Module):
    def __init__(self,
                 input_size: int, 
                 mono_size: int,
                 mono_feature,
                 exp_unit_size: Tuple = (),
                 relu_unit_size: Tuple = (),
                 conf_unit_size: Tuple = (),
                 exp_unit: ActivationLayer = ExpUnit,
                 relu_unit: ActivationLayer = ReLUUnit,
                 conf_unit: ActivationLayer = ConfluenceUnit,
                 fully_connected_layer: ActivationLayer = FCLayer):

        super(ScalableMonotonicNeuralNetwork,self).__init__()
        
        self.input_size = input_size
        self.mono_size = mono_size
        self.non_mono_size = input_size - mono_size
        self.mono_feature = mono_feature
        self.non_mono_feature = list(set(list(range(input_size))).difference(mono_feature))
        self.exp_unit_size = exp_unit_size  
        self.relu_unit_size = relu_unit_size  
        self.conf_unit_size = conf_unit_size 

        self.exp_units = torch.nn.ModuleList([
            exp_unit(mono_size if i == 0 else exp_unit_size[i-1] + conf_unit_size[i-1], exp_unit_size[i])
            for i in range(len(exp_unit_size))
        ])

        self.relu_units = torch.nn.ModuleList([
            relu_unit(self.non_mono_size if i == 0 else relu_unit_size[i-1], relu_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.conf_units = torch.nn.ModuleList([
            conf_unit(self.non_mono_size if i == 0 else relu_unit_size[i-1], conf_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.fclayer = fully_connected_layer(exp_unit_size[len(exp_unit_size)-1] + conf_unit_size[len(exp_unit_size)-1] + relu_unit_size[len(relu_unit_size)-1],1)

    def forward(self,x):

        x_mono = x[:, self.mono_feature]
        x_non_mono = x[:, self.non_mono_feature]

        for i in range(len(self.exp_unit_size)):
            if i == 0 :
                exp_output = self.exp_units[i](x_mono)
                conf_output = self.conf_units[i](x_non_mono)
                relu_output = self.relu_units[i](x_non_mono)
                exp_output = torch.cat([exp_output, conf_output], dim=1)
            else :
                exp_output = self.exp_units[i](exp_output)
                conf_output = self.conf_units[i](relu_output)
                relu_output = self.relu_units[i](relu_output)
                exp_output = torch.cat([exp_output, conf_output], dim=1)

        out = self.fclayer(torch.cat([exp_output,relu_output],dim = 1)) 
        return out
    

class SameStructure(torch.nn.Module):
    def __init__(self,
                 input_size: int, 
                 mono_size: int,
                 mono_feature,
                 exp_unit_size: Tuple = (),
                 relu_unit_size: Tuple = (),
                 conf_unit_size: Tuple = (),
                 exp_unit: ActivationLayer = ReLUUnit,
                 relu_unit: ActivationLayer = ReLUUnit,
                 conf_unit: ActivationLayer = ReLUUnit,
                 fully_connected_layer: ActivationLayer = FCLayer_notexp):

        super(SameStructure,self).__init__()
        self.input_size = input_size
        self.mono_size = mono_size
        self.non_mono_size = input_size - mono_size
        self.mono_feature = mono_feature
        self.non_mono_feature = list(set(list(range(input_size))).difference(mono_feature))

        self.exp_unit_size = exp_unit_size   
        self.relu_unit_size = relu_unit_size  
        self.conf_unit_size = conf_unit_size 

        self.exp_units = torch.nn.ModuleList([
            exp_unit(mono_size if i == 0 else exp_unit_size[i-1] + conf_unit_size[i-1], exp_unit_size[i])
            for i in range(len(exp_unit_size))
        ])

        self.relu_units = torch.nn.ModuleList([
            relu_unit(self.non_mono_size if i == 0 else relu_unit_size[i-1], relu_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.conf_units = torch.nn.ModuleList([
            conf_unit(self.non_mono_size if i == 0 else relu_unit_size[i-1], conf_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.fclayer = fully_connected_layer(exp_unit_size[len(exp_unit_size)-1] + conf_unit_size[len(exp_unit_size)-1] + relu_unit_size[len(relu_unit_size)-1],1)
        
    def forward(self,x):

        x_mono = x[:, self.mono_feature]
        x_non_mono = x[:, self.non_mono_feature]

        for i in range(len(self.exp_unit_size)):
            if i == 0 :
                exp_output = self.exp_units[i](x_mono)
                conf_output = self.conf_units[i](x_non_mono)
                relu_output = self.relu_units[i](x_non_mono)
                exp_output = torch.cat([exp_output, conf_output], dim=1)
            else :
                exp_output = self.exp_units[i](exp_output)
                conf_output = self.conf_units[i](relu_output)
                relu_output = self.relu_units[i](relu_output)
                exp_output = torch.cat([exp_output, conf_output], dim=1)

        out = self.fclayer(torch.cat([exp_output,relu_output],dim = 1)) 
        return out
    


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self,
                 input_size: int, 
                 hidden_size: Tuple = (),
                 hidden_layer: ActivationLayer = ReLUUnit,
                 fully_connected_layer: ActivationLayer = FCLayer_notexp):

        super(MultiLayerPerceptron,self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size   

        self.hidden_layers = torch.nn.ModuleList([
            hidden_layer(self.input_size if i == 0 else hidden_size[i-1], hidden_size[i])
            for i in range(len(hidden_size))
        ])

        self.fclayer = fully_connected_layer(hidden_size[len(hidden_size)-1] ,1)

    def forward(self,x):

        for i in range(len(self.hidden_size)):
            if i == 0 :
                output = self.hidden_layers[i](x)
            else :
                output = self.hidden_layers[i](output)          

        out = self.fclayer(output)
        return out


