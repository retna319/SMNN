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
parser.add_argument('--bs', help='batch_size',type=int)
parser.add_argument('--lr', help='learning_rate',type=float)
parser.add_argument('--epochs', help='epoch',type=int)   

args = parser.parse_args()   

# batch_size = 128
# structure = [(128,128),(16,16),(64,64)]
# learn_rate = 0.005



structure = [(128,128),(16,16),(64,64)]#[(16,2),(2,2),(4,4)]

def train(args):
        
    DATA_PATH = os.path.join(f'{"./"}{"synthetic_111.csv"}')
    
 
    dataset = CustomDataset(csv_path= DATA_PATH)
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.7)
    test_size = dataset_size - train_size

    
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=True, drop_last=False)

    model = ScalableMonotonicNeuralNetwork(input_size = 2,
                                            mono_size = 1,
                                            mono_feature = np.asarray([1]),
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
            
            with torch.no_grad():

                for j, (inputs, targets) in enumerate(test_dataloader):
                    x = inputs
                    y = targets
                    outputs = model(x)
                    test_loss = criterion(outputs, targets)

                    end = time.time()
                    elapsed = end - start
                                
            print('Epoch [{}/{}], Batch [{}/{}], Training Loss: {:.5f}, Test Loss: {:.5f}, Elapsed time (s): {:.4f}'.format(epoch + 1, num_epochs, batch + 1, total_batch, train_loss, test_loss.item(), elapsed))

    torch.save(model, "./model.pth")  
    contour_2d(data = dataset.inp,feature_idx=np.asarray([0,1]),model = model)
      
            





if __name__ == "__main__":
    train(args)


           