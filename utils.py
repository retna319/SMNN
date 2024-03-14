import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
	def __init__(self, csv_path):
		df = pd.read_csv(csv_path)

		self.inp = df.iloc[:, :-1].values
		self.outp = df.iloc[:,-1].values.reshape(len(df),1)
	
	def __len__(self):
		return len(self.inp) 

	def __getitem__(self,idx):
		inp = torch.FloatTensor(self.inp[idx])
		outp = torch.FloatTensor(self.outp[idx])
		return inp, outp 

         
def contour_2d(data, feature_idx, model):

    
    feature_1 = data[:,feature_idx[0]]
    feature_2 = data[:,feature_idx[1]]


    x1_grid = np.linspace(0,1,100)  
    x2_grid = np.linspace(0,1,100) 
    data_idx = 20 

    X_grid = np.array(np.meshgrid(x1_grid,x2_grid)).reshape(2,100*100).T
    dt = np.repeat(data[data_idx,:], repeats= 10000, axis = 0)
    dt.resize(2,10000)
    dt = dt.transpose()
    dt[:,feature_idx[0]] = X_grid[:,0]
    dt[:,feature_idx[1]] = X_grid[:,1]

    Y = model(torch.tensor(dt,dtype=torch.float32 ))
    #Y = 1*np.sin((dt[:,0])*(25/np.pi)) + 1*((dt[:,0]-0.5)**3) + 1*np.exp(dt[:,1]) + dt[:,1]**2 # true_y
    Y = Y.reshape(100,100)


    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    #plt.title('2d contour plot')
    CS = ax.contour(x1_grid,x2_grid, Y.detach().numpy(),levels=10,linewidths=0.5,colors ='k')
    cntr = ax.contourf(x1_grid,x2_grid, Y.detach().numpy(), levels=10, cmap="RdBu_r")
    #
    label_font = {
        'fontsize': 14,
        #'fontweight': 'bold'
    }  
    #plt.colorbar(cntr)
    plt.xlabel("x",fontdict=label_font)
    plt.ylabel("y",fontdict=label_font)

    ax.clabel(CS, inline=1, fontsize=8, colors ='k')
    plt.savefig('./contour.png', bbox_inches='tight',dpi=200)
    #plt.show()




