import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np




class BlogFeedbackdataLoader(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path,header = None)
        df = df.dropna(axis=0)
                

        self.inp = df.iloc[:,0:280].values
        self.outp = df.iloc[:,280].values.reshape(len(df),1)

    def __len__(self):
        return len(self.inp) 

    def __getitem__(self,idx):
        inp = torch.FloatTensor(self.inp[idx])
        outp = torch.FloatTensor(self.outp[idx])
        return inp, outp 

