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
from blogfeedback_loader import *
from SMNN import *
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)


# parser 
parser = argparse.ArgumentParser(description='train_SMNN')   
parser.add_argument('--bs', help='batch_size',type=int)
parser.add_argument('--lr', help='learning_rate',type=float)
parser.add_argument('--epochs', help='epoch',type=int)   

args = parser.parse_args()   



structure = [(16,8),(2,2),(2,4)]

def train(args):
        
    DATA_PATH_train = os.path.join(f'{"./"}{"train.csv"}')
    DATA_PATH_test = os.path.join(f'{"./"}{"test.csv"}')
    train_dataset = BlogFeedbackdataLoader(csv_path= DATA_PATH_train)
    test_dataset = BlogFeedbackdataLoader(csv_path= DATA_PATH_test)
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=False)

    model = ScalableMonotonicNeuralNetwork(input_size = 280,
                                            mono_size = 8,
                                            mono_feature = np.asarray([51,52,53,54,56,57,58,59]),
                                            exp_unit_size = structure[0],
                                            relu_unit_size = structure[1],
                                            conf_unit_size = structure[2])
    
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
    min_rmse = 100000

    start = time.time()
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
        with torch.no_grad():

                for j,(inputs, targets) in enumerate(test_dataloader):
                    x = inputs
                    y = targets
                    outputs = model(x)
                    test_loss = criterion(outputs, targets)
        
                    pred_y = outputs
                    true_y = targets

                    mse = criterion(pred_y, true_y)
                    rmse = mean_squared_error(pred_y, true_y,squared=False)

                    if rmse <= min_rmse:
                         torch.save(model, "./model.pth")    
                    min_rmse = min(min_rmse, rmse)

                    end = time.time()
                    elapsed = end - start       
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Training Loss: {:.5f}, Test Loss: {:.5f}, Test rmse: {:.5f}, Elapsed time (s): {:.4f}'.format(epoch + 1, num_epochs, batch + 1, total_batch, train_loss, test_loss.item(), min_rmse, elapsed))

    #torch.save(model, "./model.pth")       
            





if __name__ == "__main__":
    train(args)

           