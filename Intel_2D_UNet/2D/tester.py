# test_data_loader.py
from dataloader import DatasetGenerator, get_decathlon_filelist
import numpy as np
def sanity_check(data_loader):
    for img, mask in data_loader:
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Unique values in mask: {np.unique(mask)}")
        break  # Only check the first batch for quick validation

data_path = "/home/fahim/nnUNet/nnUNet_raw/Dataset111"
file_list = get_decathlon_filelist(data_path=data_path)

data_loader = DatasetGenerator(file_list)

# Run a sanity check
sanity_check(data_loader)
