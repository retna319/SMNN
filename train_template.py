import os
import random
import time
import numpy as np
import torch
import argparse
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
parser.add_argument('--data', help='dataset')   
parser.add_argument('--bs', help='batch_size',type=int)
parser.add_argument('--lr', help='learning_rate',type=float)
parser.add_argument('--epochs', help='epoch',type = int)   

args = parser.parse_args()   




structure = [(128,64),(16,16),(64,32)]
def train(args):
        
    DATA_PATH = os.path.join(args.data)
    
    dataset = CustomDataset(csv_path= DATA_PATH)
    
    train_dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=False)
    

    model = ScalableMonotonicNeuralNetwork(input_size = 7, # total input_size
                                            mono_size = 3, # number of monotone features
                                            mono_feature = np.asarray([1,2,3]), # position of monotone features
                                            exp_unit_size = structure[0], # size of exp_units per each layer ex) (16,8) or (16,10,15)....
                                            relu_unit_size = structure[1], # size of relu_units per each layer
                                            conf_unit_size = structure[2]) # size of conf_units per each layer
    #depth for each types of units must be equal
    
    # number of parameter
    param_amount = 0
    for p in model.named_parameters():
        #print(p[0], p[1].numel())
        param_amount += p[1].numel()
    print('total param amount:', param_amount)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= float(args.lr))


    # model train 
    num_epochs = int(args.epochs)
    total_batch = len(train_dataloader)
    model.train()

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

        if (epoch+1) % 10 == 0:
            
            end = time.time()
            elapsed = end - start                     
            print('Epoch [{}/{}], Batch [{}/{}], Training Loss: {:.5f}, Elapsed time (s): {:.4f}'.format(epoch + 1, num_epochs, batch + 1, total_batch, train_loss, elapsed))
                




if __name__ == "__main__":
    train(args)
