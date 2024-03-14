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
from compas_loader import *
from SMNN import *
from sklearn.metrics import (mean_absolute_error, accuracy_score,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)


# parser 
parser = argparse.ArgumentParser(description='train_SMNN')   
parser.add_argument('--bs', help='batch_size',type=int)
parser.add_argument('--lr', help='learning_rate',type=float)
parser.add_argument('--epochs', help='epoch',type=int)   

args = parser.parse_args()   



structure = [(32,32),(16,16),(16,16)]

def train(args):
        
    DATA_PATH_train = os.path.join(f'{"./"}{"train.csv"}')
    DATA_PATH_test = os.path.join(f'{"./"}{"test.csv"}')
    train_dataset = COMPASdataLoader(csv_path= DATA_PATH_train)
    test_dataset = COMPASdataLoader(csv_path= DATA_PATH_test)
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=False)

    model = ScalableMonotonicNeuralNetwork(input_size = 13,
                                            mono_size = 4,
                                            mono_feature = np.asarray([0,1,2,3]),
                                            exp_unit_size = structure[0],
                                            relu_unit_size = structure[1],
                                            conf_unit_size = structure[2])
    
    # number of parameter
    param_amount = 0
    for p in model.named_parameters():
        #print(p[0], p[1].numel())
        param_amount += p[1].numel()
    print('total param amount:', param_amount)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= float(args.lr))


    # model train 
    num_epochs = args.epochs
    total_batch = len(train_dataloader)
    model.train()
    max_acc = 0.000001
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

                for j, (inputs, targets) in enumerate(test_dataloader):
                    x = inputs
                    y = targets
                    outputs = model(x)
                    test_loss = criterion(outputs, targets)

                    true_y = targets.detach().numpy()
                    pred_y = outputs.detach().numpy()
                    pred_y_zero_one = np.where(pred_y > 0, 1, 0)
            
                    acc = accuracy_score(true_y,pred_y_zero_one)
                    if acc >= max_acc:
                        torch.save(model, "./model.pth")   

                    max_acc = max(max_acc, acc)        
                    end = time.time()
                    elapsed = end - start       
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Training Loss: {:.5f}, Test Loss: {:.5f}, Test acc: {:.5f}, Elapsed time (s): {:.4f}'.format(epoch + 1, num_epochs, batch + 1, total_batch, train_loss, test_loss.item(), max_acc, elapsed))

 
            





if __name__ == "__main__":
    train(args)

           