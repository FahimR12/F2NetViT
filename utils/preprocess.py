"""preprocess and prepare the dataset for training"""

import pandas as pd
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import monai
from monai import transforms

def separate_train_val_ids(json_file: str = None, fold: int = 0, phase: str = 'training'):
    """
    Separate out training ids and validation based on the fold index.
    In training, there should be about 4 folds and 1 fold should be in validation as there are 5 folds.
    
    Parameters
    ----------
    json_file: str
    fold: int
    phase: str
    
    Returns
    -------
    training: list of str
    validation: list of str
    """
    with open(json_file, 'r') as file:
        data = file.read()
    dataset = json.loads(data)
    dataset = dataset[phase]
    training = []
    validation = []
    for example in dataset:
        if example['fold'] == fold:
            patient_id = example['label'].split('/')[-2]
            validation.append(patient_id)
        else:
            patient_id = example['label'].split('/')[-2]
            training.append(patient_id)

    return training, validation
        
import pandas as pd

def insert_cases_paths_to_df(df: str, train_dir: str = None, test_dir: str = None, json_file: str = None, fold: int = 0):
    """
    Insert full case paths into the dataframe for data loading and data preparation.
    
    Parameters
    ----------
    df: str
    train_dir: str
    test_dir: str
    json_file: str
    fold: int
    
    Returns
    -------
    df: pd.DataFrame processed
    """
    # Read the CSV file
    df = pd.read_csv(df)
    print(f"Columns in DataFrame: {df.columns.tolist()}")  # Print out the column names

    # Ensure the column 'BraTS2024' exists in the DataFrame
    if 'BraTS2024' not in df.columns:
        raise KeyError(f"The column 'BraTS2024' was not found in the CSV file: {df.columns}")

    paths = []
    phase = []
    train, val = separate_train_val_ids(json_file=json_file, fold=fold)
    df = df[df['BraTS2024'].notna()]
    for _, row in df.iterrows():
        id = row["BraTS2024"]
        if id in os.listdir(train_dir):
            path = os.path.join(train_dir, id)
            print(f"Path: {path}")
            if id in train:
                type_ = "train"
            elif id in val:
                type_ = "val"
            else:
                type_ = None
        elif id in os.listdir(test_dir):
            path = os.path.join(test_dir, id)
            type_ = "test"
        paths.append(path)
        phase.append(type_)
    df['path'] = paths
    df['phase'] = phase
    print("DataFrame processing completed.")
    return df

    

def data_transforms(phase: str = 'train', roi: int = 128):
    """
    Apply data transforms to a 3D image.
    
    Parameters
    ----------
    phase: str
    roi: int
    
    Returns
    -------
    transform: transforms.Compose
    """  
    train_transform = transforms.Compose(
        [
            transforms.EnsureTyped(keys=["image", "label"]),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi, roi, roi],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi, roi, roi],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"])
        ]
    )
                
    val_transform = transforms.Compose(
        [
            transforms.EnsureTyped(keys=["image", "label"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"])
        ]
    )
    test_transform = val_transform
    transform = {'train': train_transform, 'val': val_transform, 'test': test_transform}
    return transform[phase]
