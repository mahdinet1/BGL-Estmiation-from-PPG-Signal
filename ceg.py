import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

def clarke_error_grid(ref_values, pred_values, title_string):
    # Set up the figure and axes
    
    fig, ax = plt.subplots(figsize=(12, 8),dpi=300)
    plt.rcParams["font.family"] = "Times New Roman"
    assert (len(ref_values) == len(pred_values)), "Unequal number of values."

    if max(ref_values) > 400 or max(pred_values) > 400:
        print("Warning: Values exceed normal physiological range.")
    if min(ref_values) < 0 or min(pred_values) < 0:
        print("Warning: Values below zero.")

    ax.scatter(ref_values, pred_values, marker='o', color='black', s=20)
    ax.set_title(title_string,fontsize=20, fontweight='bold',pad=25)
    ax.set_xlabel("Reference Concentration (mg/dl)",fontsize=22, fontweight='bold')
    ax.set_ylabel("Prediction Concentration (mg/dl)",fontsize=22, fontweight='bold')
    ax.set_xticks(np.arange(0, 450, 50))
    ax.set_yticks(np.arange(0, 450, 50))
    ax.set_facecolor('white')
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    ax.set_aspect('equal')

    # Colors for zones
    colors = {
        'A': '#66bb60',
        'B': '#66bb60',
        'C': '#c6ff00',
        'D': '#c62828',
        'E': '#c62828'
    }
    # Define vertices and fill polygons for each zone
    ax.fill([0, 70, 70, 400, 400, 400/1.2, 175/3, 0],
            [0, 0, 56, 400/1.24, 400, 400, 70, 70],
            color=colors['A'], alpha=0.8)
    ax.fill([70,400/1.2,290,70], [84,400,400,180], color=colors['B'], alpha=0.5)
    ax.fill([70,130,180,240,240,400,400,70], [0,0,70,70,180,180,400/1.25,56], color=colors['B'], alpha=0.5)
    ax.fill([240,400,400,240], [70,70,180,180], color=colors['D'], alpha=0.6)
    ax.fill([180,400,400,180], [0,0,70,70], color=colors['E'], alpha=0.8)
    ax.fill([130,180,180], [0,0,70], color=colors['C'], alpha=0.4)
    ax.fill([0,175/3,70,70,0], [70,70,84,180,180], color=colors['D'], alpha=0.6)
    ax.fill([0,70,70,0], [180,180,400,400], color=colors['E'], alpha=0.8)
    ax.fill([70,290,70], [180,400,400], color=colors['C'], alpha=0.4)

    # Create a color bar
    cmap = mcolors.ListedColormap([colors['A'], "#B2DCAF", colors['C'], '#DC7E7E', '#D15353'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=[0.2, 1, 2, 3, 3.8])
    cbar.set_ticklabels(['No Risk (A)', 'Low Risk (B)', 'High Risk (C)', 'Very High Risk (D)', 'Extreme Risk (E)'],fontsize=20)
    cbar.set_label('Clinical Risk')
    plt.tick_params(axis='both', labelsize=18, )
    # Save and show plot
    plt.savefig(title_string + '.png', dpi=300)
    # plt.show()

    # Collect statistics
    zone_counts = [0] * 5
    for ref, pred in zip(ref_values, pred_values):
        if (ref <= 70 and pred <= 70) or (pred <= 1.2*ref and pred >= 0.8*ref):
            zone_counts[0] += 1  # Zone A
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zone_counts[4] += 1  # Zone E
        elif ((ref >= 70 and ref <= 290) and pred >= ref + 110) or ((ref >= 130 and ref <= 180) and (pred <= (7/5)*ref - 182)):
            zone_counts[2] += 1  # Zone C
        elif (ref >= 240 and (pred >= 70 and pred <= 180)) or (ref <= 175/3 and pred <= 180 and pred >= 70) or ((ref >= 175/3 and ref <= 70) and pred >= (6/5)*ref):
            zone_counts[3] += 1  # Zone D
        else:
            zone_counts[1] += 1  # Zone B

    # Return the normalized zone counts
    return plt, [z / len(ref_values) for z in zone_counts]

# resnet_preds = np.load("results/resnet34-1scnnLstmAtt-1s.npy")
# print(resnet_preds)
# vgg16_preds = np.load("results/vgg16-1scnnLstmAtt-1s.npy")
# custom = np.load("results/cnnLstmAtt-1s.npy")
# test_data = np.load("tes.npy",allow_pickle = True)
# labels = test_data[:,1].reshape((-1,1))
# print(labels)

# plt, zone_counts = clarke_error_grid(labels,resnet_preds,"resnet34 1sec")
# print(zone_counts)
# plt, zone_counts = clarke_error_grid(labels,vgg16_preds,"vgg16 1sec")
# print(zone_counts)
# plt, zone_counts = clarke_error_grid(labels,custom,"cnnLstmAtt 1sec")
# print(zone_counts)

import torch
import torch.nn as nn

def mean_absolute_relative_difference(true, pred):
    """
    Computes the Mean Absolute Relative Difference (MARD) between the true values and predictions.

    Parameters:
    - true (torch.Tensor): The ground truth values.
    - pred (torch.Tensor): The predicted values.

    Returns:
    - float: The MARD value.
    """
    # Ensure that the true values and predictions are of the same shape
    assert true.shape == pred.shape, "True values and predictions must have the same shape."

    # Calculate the absolute differences between true values and predictions
    absolute_differences = np.abs(true - pred)

    # Calculate the relative differences (as a percentage)
    true_clamped = np.clip(true, a_min=1e-8, a_max=None)
    relative_differences = absolute_differences / true_clamped
    

    # Calculate the mean of these relative differences
    mard = np.mean(relative_differences) * 100  # Convert to percentage

    return mard

def r_squared(true, preds):
    """
    Compute the coefficient of determination (R^2) score using NumPy.

    Parameters:
    - true (numpy.ndarray): The ground truth (actual) values.
    - preds (numpy.ndarray): The predicted values from the model.

    Returns:
    - float: The R^2 score.
    """
    # Ensure that true values and predictions are of the same shape
    assert true.shape == preds.shape, "True values and predictions must have the same shape."

    # Calculate the total sum of squares (SST)
    mean_true = np.mean(true)
    total_variance = np.sum((true - mean_true) ** 2)

    # Calculate the residual sum of squares (SSR)
    residuals = true - preds
    residual_variance = np.sum(residuals ** 2)

    # Calculate R^2
    r2_score = 1 - (residual_variance / total_variance)

    return r2_score


from sklearn.metrics import mean_squared_error,mean_absolute_error
mse = mean_squared_error
MAE = mean_absolute_error

def metrics_mazandaran_data():
    data = np.load("maz_preds_cnnlstmatt.npy")
    labels =data[:,1].astype("float32")
    preds = data[:,2].astype("float32")
    mse_loss = mse(labels,preds)
    mae_loss = MAE(labels,preds)
    mard_loss = mean_absolute_relative_difference(labels,preds)
    r_squared_loss = r_squared(labels,preds)
    plot,zones = clarke_error_grid(labels,preds,"Assessment of Clinical Risk Levels for MUST Dataset Predictions cnnlstmatt")
    print("zones",zones)
    plt.figure(figsize=(10, 5))
    residuals = labels - preds

    # Plotting the residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(preds, residuals, color='blue',linewidths=3)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot for Predicted Blood Glucose Levels on MUST Dataset',fontsize=20, fontweight='bold')
    plt.xlabel('Predicted Values (mg/dL)',fontsize=20, fontweight='bold')
    plt.ylabel('Residuals (mg/dL)',fontsize=22, fontweight='bold')
    plt.tick_params(axis='both', labelsize=22, )
    # plt.savefig("Residual Plot for Predicted Blood Glucose Levels on MUST Dataset",dpi=300)
    plt.show()
    # plt.plot(x, y_actual, 'r-', label='Actual Data')  # Plot actual data for comparison

    print(f'vgg16 metrics: mse:{mse_loss} mae:{mae_loss} MARD:{mard_loss} R2:{r_squared_loss} RMSE:{mse_loss ** 0.5}')


metrics_mazandaran_data()


# print(labels.shape,resnet_preds.shape)
# mse_loss_resnet = mse(labels, resnet_preds)
# mse_loss_vgg = mse(labels, vgg16_preds)
# mse_loss_custom = mse(labels, custom)



# # compute the mean absolute error
# mae_loss_resnet = MAE(labels, resnet_preds)
# mae_loss_vgg = MAE(labels, vgg16_preds)
# mae_loss_custom = MAE(labels, custom)


# mard_loss_resnet = mean_absolute_relative_difference(labels,resnet_preds)
# mard_loss_vgg = mean_absolute_relative_difference(labels,vgg16_preds)
# mard_loss_custom = mean_absolute_relative_difference(labels,custom)

# r2_resnet = r_squared(labels,resnet_preds)
# r2_vgg = r_squared(labels,vgg16_preds)
# r2_custom = r_squared(labels,custom)

# print(f'resnet34 metrics: mse:{mse_loss_resnet} mae:{mae_loss_resnet} MARD:{mard_loss_resnet} R2:{r2_resnet} RMSE:{mse_loss_resnet ** 0.5}')
# print(f'VGG16 metrics: mse:{mse_loss_vgg} mae:{mae_loss_vgg} MARD:{mard_loss_vgg} R2:{r2_vgg} RMSE:{mse_loss_vgg ** 0.5}')
# print(f'cnnLstmAtt metrics: mse:{mse_loss_custom} mae:{mae_loss_custom} MARD:{mard_loss_custom} R2:{r2_custom} RMSE:{mse_loss_custom ** 0.5}')
