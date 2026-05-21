import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from load_data import train_loader,val_loader
from vgg16 import VGG16
from cnnLstmAtt import CustomModel
# from cnnTransformer import combined_model
# from resnet import resnet34
from torchsummary import summary
# Define the model
model = CustomModel()
# model = combined_model
# model = resnet34()

model.to('cuda')
# summary(model, (1,1000))
# print(model)  # Print model summary
model_name = "vgg16"
checkpoint_path = 'vgg16_10s_checkpoint.pth'

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Define early stopping parameters
patience = 15
early_stopping_counter = 0
best_val_loss = float('inf')
start_epoch = 0
best_model_weights = None
train_loss_array = []
val_loss_array = []
from tqdm import tqdm
epochs = 100

# Check if checkpoint exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    train_loss_array = list(checkpoint['train_loss_array'])
    val_loss_array = list(checkpoint['val_loss_array'])
    best_val_loss = min(val_loss_array)
# Training loop
for epoch in range(start_epoch, epochs):  # Adjust as needed
    # Training
    model.train()
    train_loss = 0.0
    print(f'Epoch {epoch + 1}/{100}')
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (inputs, labels) in pbar:
        optimizer.zero_grad()
        inputs = inputs.permute(0, 2, 1).to('cuda')
        labels = labels.reshape((-1,1))
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        pbar.set_description(f'Epoch {epoch + 1}/{epochs}, Step {step + 1}/{len(train_loader)}, Train Loss: {loss.item():.6f}')

    train_loss /= len(train_loader.dataset)
    train_loss_array.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.permute(0, 2, 1).to('cuda')
            labels = labels.reshape((-1,1))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_array.append(val_loss)
    print(f'Epoch {epoch+1}/{100}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    print("****************************************")
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss_array': np.array(train_loss_array),
        'val_loss_array': np.array(val_loss_array)
    }, checkpoint_path)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model weights
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), 'vgg16-10s.pth')
    np.save("vgg16-10s-train-loss",train_loss_array)
    np.save("vgg16-10s-val-loss",val_loss_array)