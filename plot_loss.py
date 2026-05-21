import matplotlib.pyplot as plt
import matplotlib
import numpy as np


resnet_train = np.load("C:\\Users\\mahdi\\Documents\\models-results\\1s\\resnet\\resnet34-train-loss.npy")
resnet_val = np.load("C:\\Users\\mahdi\\Documents\\models-results\\1s\\resnet\\resnet34-val-loss.npy")

vgg_train = np.load("C:\\Users\\mahdi\\Documents\\models-results\\1s\\vgg\\vgg16-1s-train-loss.npy")
vgg_val = np.load("C:\\Users\\mahdi\\Documents\\models-results\\1s\\vgg\\vgg16-1s-val-loss.npy")

cnnLstm_train = np.load("C:\\Users\\mahdi\\Documents\\models-results\\1s\\cnnLstm\\cnnLstmAtt-1s-train-loss.npy")
cnnLstm_val = np.load("C:\\Users\\mahdi\\Documents\\models-results\\1s\\cnnLstm\\cnnLstmAtt-1s-val-loss.npy")
print(vgg_val,vgg_train)
plt.rcParams["font.family"] = "Times New Roman"
# Set the font to Times New Roman
# matplotlib.rc('font', family='Times New Roman')

# Define data - assuming these variables are defined elsewhere with the respective loss data
# resnet_train, resnet_val, vgg_train, vgg_val

plt.figure(figsize=(16, 8))  # Sets the figure size for better visibility

# Plot for ResNet model
plt.plot(resnet_train, label='ResNet34 1s Training Loss', color='navy', linestyle='-', marker='o', markersize=5)
plt.plot(resnet_val, label='ResNet34 1s Validation Loss', color='crimson', linestyle='--', marker='x', markersize=5)

# Plot for VGG model
plt.plot(vgg_train, label='VGG16 1s Training Loss', color='black', linestyle='-.', marker='s', markersize=5)
plt.plot(vgg_val, label='VGG16 1s Validation Loss', color='green', linestyle=':', marker='^', markersize=5)

plt.plot(cnnLstm_train, label='CNN-LSTM-ATT 1s Training Loss',  markersize=5)
plt.plot(cnnLstm_val, label='CNN-LSTM-ATT 1s Validation Loss', markersize=5)

# Adding a title and labels with custom fonts
plt.title('Training and Validation Loss Across 1sec Models', fontsize=20, fontweight='bold')
plt.xlabel('Epochs', fontsize=20, fontweight='bold')
plt.ylabel('Loss', fontsize=20, fontweight='bold')

# Enhance the legend to be more informative
plt.legend(loc='upper right', fontsize=20, frameon=True, shadow=True)
plt.tick_params(axis='both', labelsize=18, )
# Add a grid for easier reading of the plot, make it lighter and less obtrusive
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
# Optionally, save the plot to a file in high resolution
plt.savefig('model_comparison_loss_plot.png', dpi=300)  # Save as high-resolution
# Show the plot on screen
plt.show()


