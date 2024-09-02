import json
import os
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np

# Define paths
brats_data_dir = '/home/fahim/nnUNet/nnUNet_raw/Dataset111_BraTS2024'
images_dir = os.path.join(brats_data_dir, "imagesTr")
labels_dir = os.path.join(brats_data_dir, "labelsTr")

# Function to convert labels to binary WT (whole tumor) vs background
def convert_labels_to_binary_WT(in_file: str, out_file: str) -> None:
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img).astype(np.int32)

    # Print the shape of the original label
    print(f'Original label shape for {in_file}: {img_npy.shape}')
    
    # Convert labels to binary: WT (1) vs background (0)
    seg_new = np.zeros_like(img_npy)
    seg_new[(img_npy == 1) | (img_npy == 2) | (img_npy == 3)] = 1

    # Print the shape of the new binary label
    print(f'Binary label shape for {out_file}: {seg_new.shape}')
    
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

# Get all FLAIR images (ending with '_0003.nii.gz')
flair_images = [f for f in os.listdir(images_dir) if f.endswith('_0003.nii.gz')]
flair_images = sorted(flair_images)

# Prepare image and label pairs
dataset = []
for flair_image in flair_images:
    # Extract base ID from the FLAIR image filename
    base_id = flair_image.replace('_0003.nii.gz', '')

    # Construct corresponding label filename
    label_filename = f"{base_id}.nii.gz"

    # Define new binary label output path
    binary_label_output = os.path.join(labels_dir, f"binary_{label_filename}")
    
    # Convert labels to binary WT vs background
    if os.path.exists(os.path.join(labels_dir, label_filename)):
        print(f'Converting label for {label_filename}...')
        convert_labels_to_binary_WT(
            os.path.join(labels_dir, label_filename),
            binary_label_output
        )

        # Add to dataset list if the label file exists
        dataset.append({
            "image": f"./imagesTr/{flair_image}",
            "label": f"./labelsTr/binary_{label_filename}"
        })

# Split dataset into training and validation
train_files = dataset

# Construct the dataset dictionary for JSON
dataset_dict = {
    "name": "BRATS",
    "description": "Gliomas segmentation tumor and edema in brain images using FLAIR modality for WT segmentation",
    "reference": "https://www.med.upenn.edu/sbia/brats2017.html",
    "licence": "CC-BY-SA 4.0",
    "release": "2.0 04/05/2018",
    "tensorImageSize": "4D",
    "modality": {
        "0": "FLAIR"
    },
    "labels": {
        "0": "background",
        "1": "whole tumor"
    },
    "numTraining": len(train_files),
    "numTest": 0,
    "training": train_files,
    "test": []
}

# Save the JSON file
dataset_json_path = os.path.join(brats_data_dir, "bin_dataset.json")
with open(dataset_json_path, "w") as json_file:
    json.dump(dataset_dict, json_file, indent=4)

print(f"dataset.json created at: {dataset_json_path}")
