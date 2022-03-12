import os
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

class CheXpertDataset(data.Dataset):
    
    def __init__(self, label_strategy, version='small', mode='train', path='/gpu-data2/jpik', transform=None):

        # Change the path accordingly
        self.path = path
        self.transform = transform
        self.mode = mode
        self.strategy = label_strategy

        self.conditions = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", 
                           "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", 
                           "Fracture", "Support Devices"]

        self.attributes = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]
        
        self.df = pd.read_csv(os.path.join(self.path, 'CheXpert-v1.0-{}/{}.csv'.format(version, mode)))
        
        # Replace NaN condition values with zeros
        self.df = self.df.fillna(value=dict.fromkeys(self.conditions, 0))

        # Uncertain label replacement
        if self.mode == 'train' and self.strategy == 'U-Zeros':
            self.df = self.df.replace(dict.fromkeys(self.conditions, -1), 0)
        elif self.mode == 'train' and self.strategy == 'U-Ones':
            self.df = self.df.replace(dict.fromkeys(self.conditions, -1), 1)
                
    def __getitem__(self, index):
                
        conditions = self.df.iloc[index][self.conditions]

        fname = os.path.join(self.path, self.df.iloc[index]['Path'])
        
        img = Image.open(os.path.join(self.path, fname)).convert("RGB")
        
        if self.transform is None:
            process_img = img
        else:
            process_img = self.transform(img)
        
        return process_img, torch.tensor(conditions).float()
    
    def __len__(self):
        return len(self.df)    
