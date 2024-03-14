import os
import random
import sys
import time
import numpy as np
import torch
import argparse

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from pickle import load
from utils import *
from autompg_loader import *
from SMNN import *

from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)




def test():
        
    DATA_PATH_test = os.path.join(f'{"./"}{"test.csv"}')
    test_dataset = AutompgdataLoader(csv_path= DATA_PATH_test)
    print(f"Testing Data Size : {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=False)

    model = torch.load("./model.pth")

    # number of parameter
    param_amount = 0
    for p in model.named_parameters():
        #print(p[0], p[1].numel())
        param_amount += p[1].numel()
    print('total param amount:', param_amount)

    criterion = torch.nn.MSELoss()

    # predict
    with torch.no_grad():

        for j,(inputs, targets) in enumerate(test_dataloader):
            x = inputs
            y = targets
            outputs = model(x)
            test_loss = criterion(outputs, targets)
                  
            load_scaler = load(open(f'{"./"}{"scaler.pkl"}', 'rb'))
            pred_y = outputs*(load_scaler.data_max_[0] - load_scaler.data_min_[0]) + load_scaler.data_min_[0]
            true_y = targets*(load_scaler.data_max_[0] - load_scaler.data_min_[0]) + load_scaler.data_min_[0]
            mse = criterion(pred_y, true_y)    

    print('Test Loss: {:.5f}, Test mse: {:.5f}'.format(test_loss.item(), mse))

   
            
if __name__ == "__main__":
    test()

