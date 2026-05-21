import os
import shutil
import random
data = os.listdir("/segments")
random.shuffle(data)

train_files = data[:int(len(data) * 0.7)]
val_files = data[int(len(data) * 0.7):int(len(data) * 0.7) + int(len(data) * 0.3)]
test_files = data[int(len(data) * 0.7)+  int(len(data) * 0.3):]
# Function to write file paths to a text file
def write_to_file(file_list, filename):
    with open(filename, 'w') as file:
        for path in file_list:
            file.write(f"/segments/{path}\n")

# Writing lists to files
write_to_file(train_files, 'train_files.txt')
write_to_file(val_files, 'val_files.txt')
write_to_file(test_files, 'test_files.txt')

