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


# parser 
parser = argparse.ArgumentParser(description='train_SMNN')   
parser.add_argument('--model', help='model')
parser.add_argument('--bs', help='batch_size',type=int)
parser.add_argument('--lr', help='learning_rate',type=float)
parser.add_argument('--epochs', help='epoch',type=int)   

args = parser.parse_args()   

# batch_size = 128
# structure = [(128,128),(16,16),(64,64)]
# learn_rate = 0.005



structure = [(128,128),(16,16),(64,64)]

def train(args):
        
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=False)
    denoised_dataloader = DataLoader(denoised_dataset, batch_size=len(denoised_dataset), shuffle=True, drop_last=False)

    if args.model == "SMNN":
        model = ScalableMonotonicNeuralNetwork(input_size = 5,
                                                mono_size = 2,
                                                mono_feature = np.asarray([3,4]),
                                                exp_unit_size = structure[0],
                                                relu_unit_size = structure[1],
                                                conf_unit_size = structure[2])    
    if args.model == "SAME":    
        model = SameStructure(input_size = 5,
                                mono_size = 2,
                                mono_feature = np.asarray([3,4]),
                                exp_unit_size = structure[0],
                                relu_unit_size = structure[1],
                                conf_unit_size = structure[2])
    if args.model == "MLP":    
        model = MultiLayerPerceptron(input_size = 5,
                                     hidden_size = (208,208))
    
    # number of parameter
    param_amount = 0
    for p in model.named_parameters():
        #print(p[0], p[1].numel())
        param_amount += p[1].numel()
    print('total param amount:', param_amount)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= float(args.lr))


    # model train 
    num_epochs = args.epochs
    total_batch = len(train_dataloader)
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch, (inputs, targets) in enumerate(train_dataloader):


            outputs = model(inputs) 
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss = train_loss/total_batch

        if (epoch+1) % 10 == 0:
            
            with torch.no_grad():

                for j, (inputs, targets) in enumerate(test_dataloader):
                    x = inputs
                    y = targets
                    outputs = model(x)
                    test_loss = criterion(outputs, targets)

                for k, (d_inputs, d_targets) in enumerate(denoised_dataloader):
                    d_x = d_inputs
                    d_y = d_targets
                    d_outputs = model(d_x)
                    d_test_loss = criterion(d_outputs, d_targets)
                                
            print('Epoch [{}/{}], Batch [{}/{}], Training Loss: {:.3f}, Test Loss: {:.3f}, Denoised Test Loss: {:.3f}'.format(epoch + 1, num_epochs, batch + 1, total_batch, train_loss, test_loss.item(), d_test_loss.item()))

    if args.model == "SMNN":
        torch.save(model, "./model_SMNN.pth")
    if args.model == "SAME":    
        torch.save(model, "./model_SAME.pth")    
    if args.model == "MLP":    
        torch.save(model, "./model_MLP.pth")    
          
            





if __name__ == "__main__":
    train(args)

           