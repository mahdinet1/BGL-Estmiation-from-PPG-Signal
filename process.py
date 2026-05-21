
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import json

class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        ppgs, label = load_data(file_path)
        return ppgs, label



def coeff_determination(y_true, y_pred):
    SS_res = torch.sum((y_true - y_pred) ** 2)
    SS_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    epsilon = torch.tensor(1e-07)  # Small value to avoid division by zero
    return 1 - SS_res / (SS_tot + epsilon)

def load_data(file_path):
    # Load your data here, adjust as needed
    
    # label_path = "" + file_path.split('/')[-1] + "/label_id_" + str(id) + '.txt'
    
    # data = np.load(file_path, allow_pickle=True)
    with open(file_path) as f:
        data = json.load(f)
        signal = np.array(data['signal'])
        ten_sec_segments = signal.reshape((-1,1000))
    ppgs = torch.tensor(scale_to_range(np.vstack(ten_sec_segments)), dtype=torch.float32)
    label = torch.tensor(float(data['bgl']), dtype=torch.float32)
    # with open(label_path,'r') as f:
    #   label = float(f.read())
    return ppgs, label

# Define a function to scale data to the range (-1, 1)
def scale_to_range(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Define a function to filter files
def func(filepath):
    if filepath.endswith('.npy'):
        return True
    return False



