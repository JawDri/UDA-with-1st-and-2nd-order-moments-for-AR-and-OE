import torch
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset




class PytorchDataSet(Dataset):
    
    def __init__(self, df):
        FEATURES = list(i for i in df.columns if i!= 'labels')
        TARGET = "labels"

        from sklearn.preprocessing import StandardScaler
        Normarizescaler = StandardScaler()
        Normarizescaler.fit(np.array(df[FEATURES]))
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        
        self.train_X = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.train_X = Normarizescaler.transform(self.train_X)
        
        
        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        return self.train_X[idx], self.train_Y[idx]






def get_loader(name_dataset, batch_size, train=True):
    
    
    

    # Computed with compute_mean_std.py
    if name_dataset == 'source':
      Source_train = pd.read_csv("/content/drive/MyDrive/DCORAL/data/Source_train.csv")
      dataset = PytorchDataSet(Source_train)
    elif name_dataset == 'target':
      Target_train = pd.read_csv("/content/drive/MyDrive/DCORAL/data/Target_train.csv")
      dataset = PytorchDataSet(Target_train)
    elif name_dataset == 'target_eval':
      Target_test = pd.read_csv("/content/drive/MyDrive/DCORAL/data/Target_test.csv")
      dataset = PytorchDataSet(Target_test)
      
    elif name_dataset == 'source_eval':
      Source_test = pd.read_csv("/content/drive/MyDrive/DCORAL/data/Source_test.csv")
      dataset = PytorchDataSet(Source_test)

    
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=train)
    return dataset_loader

