import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
# from load_data import test_loader
from vgg16 import VGG16
from cnnLstmAtt import CustomModel
from resnet import resnet34
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

res_model = resnet34()
# res_model.load_state_dict(torch.load("C:\\Users\\mahdi\\Documents\\models-results\\10s\\resnet34\\resnet34-10s.pth",map_location=torch.device('cpu')))
res_model.to('cpu')

vgg_model = VGG16()
# vgg_model.load_state_dict(torch.load("vgg16-10s.pth",map_location=torch.device('cpu')))
vgg_model.to('cpu')

# model_name = "cnnLstmAtt-1s"
model = CustomModel()
# model.load_state_dict(torch.load("C:\\Users\\mahdi\\Documents\\models-results\\10s\\cnnlstm\\cnnLstmAtt-10s.pth",map_location=torch.device('cpu')))
# model.to('cpu')

print(summary(res_model,(1,1000)))
print(summary(vgg_model,(1,1000)))
print(summary(model,(1,1000)))
exit()
# model.to('cuda:0')
# model.fuse()
from tqdm import tqdm



criterion = nn.MSELoss()

def scale_to_range(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data
print('loading test.npy')
# ppgs = np.load("tes.npy",allow_pickle = True)[:,2]
print('loading ppgs')

# ppgs = [scale_to_range(ppg.reshape((-1,1))).reshape((1,-1)) for ppg in tqdm(ppgs)]

# ppgs = np.stack(ppgs,axis=0)

# ppgs = torch.tensor(ppgs, dtype=torch.float32)
# print(ppgs.shape)
# ppgs_dataset = TensorDataset(ppgs)
# batch_size = 128  # You can adjust the batch size depending on your GPU's memory
# test_loader = DataLoader(ppgs_dataset, batch_size=batch_size, shuffle=False)
# # print(mse_loss,mae_loss,mard_loss,r2)

cnn_preds = []
vgg_preds = []
res_preds = []

model.eval()
res_model.eval()
vgg_model.eval()
with torch.no_grad():
    # pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for inputs, labels in test_loader:
        # print(inputs)
        inputs = inputs.permute(0, 2, 1).to('cuda')
        labels = labels.reshape((-1,1))
        outputs = model(inputs)
        res_outs = res_model(inputs)
        vgg_out = vgg_model(inputs)
        # loss = criterion(outputs, labels)
       
        cnn_preds.extend((labels,outputs.cpu().tolist()))
        vgg_preds.extend((labels,vgg_out.cpu().tolist()))
        res_preds.extend((labels,res_outs.cpu().tolist()))
        
        
        # pbar.set_description(f' Step {step + 1}/{len(test_loader)}')
        # print(cnn_preds)
        # exit()
    np.save("./results/cnnLstmAtt-10s" ,np.array(cnn_preds))
    np.save("./results/resnet34-10s"  ,np.array(res_preds))
    np.save("./results/vgg16-10s"  ,np.array(vgg_preds))