import os
from process import CustomDataset,DataLoader


# Define train and validation file paths
# train_files = os.listdir("train/")[:]  # List of file paths for training data
# val_files = os.listdir("val/")[:]  # List of file paths for validation data
# base_dir = ""
# train_files = [base_dir +'train/' +i for i in train_files]
# val_files = [base_dir +'val/' +i for i in val_files]


def read_file_to_list(filename):
    with open(filename, 'r') as file:
        # Read all lines and strip newline characters, then return as a list
        return [line.strip() for line in file]

# Reading the lists from files
train_files = read_file_to_list('train_files.txt')
val_files = read_file_to_list('val_files.txt')
test_files = read_file_to_list('test_files.txt')

# Create custom datasets
train_dataset = CustomDataset(train_files)
val_dataset = CustomDataset(val_files)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

