import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        e = F.relu(torch.matmul(x, self.W) + self.b)
        a = F.softmax(e, dim=1)
        output = x * a
        return torch.sum(output, dim=1)

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.bigru = nn.LSTM(input_size=11, hidden_size=16, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.attention = AttentionLayer(input_dim=32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x, (_,_) = self.bigru(x)
        x = self.dropout(x)
        x = self.attention(x)
        x = self.fc(x)
        return x
