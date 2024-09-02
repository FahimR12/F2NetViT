import os
import csv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configures.my_config import MyConfig

conf_base_path = '/home/fahim/F2NetViT/conf'
configs = MyConfig(conf_base_path)
# Define directories and file lists

train_dir = configs.Configs.full_paths["train_path"]
valid_dir = configs.Configs.full_paths["validation_path"]
train_list_file = '/home/fahim/F2NetViT/Data/train_list.txt'
valid_list_file = '/home/fahim/F2NetViT/Data/valid_list.txt'
output_csv = 'metadata.csv'

# Define file modalities
modalities = ['t2f', 'seg', 't1n', 't2w', 't1c']

# Function to read patient IDs from a file
def read_patient_ids(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

# Get patient IDs for training and validation
train_patient_ids = read_patient_ids(train_list_file)
valid_patient_ids = read_patient_ids(valid_list_file)

# Create metadata list
metadata = []

# Helper function to add entries to metadata
def add_entries(patient_ids, directory, split):
    for patient_id in patient_ids:
        for modality in modalities:
            filename = f"{patient_id}-{modality}.nii.gz"
            filepath = os.path.join(directory, patient_id, filename)
            metadata.append({
                'BraTS2024': filepath,
                'label': modality,
                'split': split,
                'patient_id': patient_id
            })

# Add entries for training and validation datasets
add_entries(train_patient_ids, train_dir, 'train')
add_entries(valid_patient_ids, valid_dir, 'valid')

# Write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['BraTS2024', 'label', 'split', 'patient_id']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(metadata)

print(f"Metadata file created at {output_csv}")
