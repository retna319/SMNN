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
from SMNN import *

from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)




def test():
        
    DATA_PATH_train = os.path.join(f'{"./"}{"noise10_train.csv"}')
    DATA_PATH_test = os.path.join(f'{"./"}{"noise10_test.csv"}')
    DATA_PATH_denoised = os.path.join(f'{"./"}{"noise10_denoised.csv"}')
 
    train_dataset = CustomDataset(csv_path= DATA_PATH_train)
    test_dataset = CustomDataset(csv_path= DATA_PATH_test)
    denoised_dataset = CustomDataset(csv_path= DATA_PATH_denoised)

    
    train_size = len(train_dataset.inp)+1
    test_size = len(test_dataset.inp)+1
    #denoised_size = len(denoised_dataset.inp)

    print(f"Training Data Size : {train_size}")
    print(f"Testing Data Size : {test_size}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=True, drop_last=False)
    denoised_dataloader = DataLoader(denoised_dataset, batch_size=test_size, shuffle=True, drop_last=False)

    model_SMNN = torch.load("./model_SMNN.pth")
    model_SAME = torch.load("./model_SAME.pth")
    model_MLP = torch.load("./model_MLP.pth")



    criterion = torch.nn.MSELoss()

    # predict
    with torch.no_grad():

        for i, (inputs, targets) in enumerate(train_dataloader):
            x = inputs

            outputs_train_SMNN = model_SMNN(x)
            outputs_train_SAME = model_SAME(x)
            outputs_train_MLP = model_MLP(x)
            train_loss_SMNN = criterion(outputs_train_SMNN, targets)
            train_loss_SAME = criterion(outputs_train_SAME, targets)
            train_loss_MLP = criterion(outputs_train_MLP, targets)

        for j, (inputs, targets) in enumerate(test_dataloader):
            x = inputs
            outputs_test_SMNN = model_SMNN(x)
            outputs_test_SAME = model_SAME(x)
            outputs_test_MLP = model_MLP(x)
            test_loss_SMNN = criterion(outputs_test_SMNN, targets)
            test_loss_SAME = criterion(outputs_test_SAME, targets)
            test_loss_MLP = criterion(outputs_test_MLP, targets)

        for k, (d_inputs, d_targets) in enumerate(denoised_dataloader):
            d_x = d_inputs

            outputs_denoised_SMNN = model_SMNN(d_x)
            outputs_denoised_SAME = model_SAME(d_x)
            outputs_denoised_MLP = model_MLP(d_x)
            denoised_loss_SMNN = criterion(outputs_denoised_SMNN, d_targets)
            denoised_loss_SAME = criterion(outputs_denoised_SAME, d_targets)
            denoised_loss_MLP = criterion(outputs_denoised_MLP, d_targets)

    print('(SMNN) Train Loss: {:.3f}, Test Loss: {:.3f}, Denoised Test Loss: {:.3f}'.format(train_loss_SMNN.item(), 
                                                                                            test_loss_SMNN.item(), 
                                                                                            denoised_loss_SMNN.item()))
    print('(SAME) Train Loss: {:.3f}, Test Loss: {:.3f}, Denoised Test Loss: {:.3f}'.format(train_loss_SAME.item(), 
                                                                                            test_loss_SAME.item(), 
                                                                                            denoised_loss_SAME.item()))
    print(' (MLP) Train Loss: {:.3f}, Test Loss: {:.3f}, Denoised Test Loss: {:.3f}'.format(train_loss_MLP.item(), 
                                                                                            test_loss_MLP.item(), 
                                                                                            denoised_loss_MLP.item()))

   
            





if __name__ == "__main__":
    test()

