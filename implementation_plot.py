# import matplotlib.pyplot as plt
# import numpy as np
# plt.rcParams["font.family"] = "Times New Roman"
# # Data
# categories = ["Base Model", "Pruned Model", "PQAT Model"]
# values = [755, 810, 795]
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Assigning distinct colors for each bar

# # Creating the bar chart
# fig, ax = plt.subplots()
# # Setting the width of the bars
# bar_width = 0.5
# bars = ax.bar(categories, values, color=colors,width=bar_width)

# # Adding numbers on top of the bars
# for bar in bars:
#     yval = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

# # Adding titles and labels
# # ax.set_title('MSE Loss of 3 Methods of Optimizing Models for Embedded Devices', fontsize=14, fontweight='bold')
# ax.set_ylabel('MSE Loss', fontsize=12)
# ax.set_xlabel('Model Type', fontsize=12)

# # Removing the top and right spines for a cleaner look
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Adding grid lines for better readability
# ax.yaxis.grid(True, linestyle='--', alpha=0.7)
# plt.savefig("MSE Loss of 3 Methods of Optimizing Models for Embedded Devices.png",dpi=300)
# # Display the plot
# plt.show()






import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
# Data
categories = ["Base Model", "Pruned Model", "PQAT Model"]
values = [75, 25, 7]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Assigning distinct colors for each bar

# Creating the bar chart
fig, ax = plt.subplots(figsize=(10,8),dpi=400)
# Setting the width of the bars
bar_width = 0.5
bars = ax.bar(categories, values, color=colors,width=bar_width)

# Adding numbers on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom',fontsize=18)

# Adding titles and labels
# ax.set_title('MSE Loss of 3 Methods of Optimizing Models for Embedded Devices', fontsize=14, fontweight='bold')
ax.set_ylabel('Model Size (MB)', fontsize=20, fontweight='bold')
ax.set_xlabel('Model Type', fontsize=20, fontweight='bold')

# Removing the top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adding grid lines for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.savefig("Model_size.png",dpi=1200)
# Display the plot
plt.show()
