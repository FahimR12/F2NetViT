# SWINUNetR Model

This repository implements the `SWINUNetR` architecture for BraTS 2024 tumour segmentation tasks. The architecture was selected due to issues with the original model and optimizations in this implementation. Below are key changes and features included in this project.

## Model Overview

This project implements the `SWINUNetR` model to handle the BraTS 2024 dataset. The following features and changes have been incorporated into this model:

- **Error Handling**: 
  - Improved handling for file-not-found errors.
  - Added functionality for correcting file paths dynamically.
  
- **Data Augmentation Enhancements**: 
  - Extended data augmentation techniques for better generalization.

- **Performance Metrics**:
  - Includes calculation of Dice coefficient score.
  - Implements Intersection over Union (IoU) metrics.

### Changes Made

- Switched to `SWINUNetR` architecture due to memory issues with the original model.
- Optimized training for mixed-precision to reduce memory usage.
- Integrated comprehensive logging for better training/validation tracking.

## Training Pipeline

The complete training and validation pipeline for the BraTS 2024 tumour segmentation task is implemented (USE GPU with VRAM larger than 24GB). Below is a summary of the key steps:

- **Training Script**:
  - Loads the BraTS 2024 dataset and preprocesses the input images and masks.
  - Uses data augmentation techniques for training robustness.
  - Supports mixed precision training to optimize memory usage.
  - Allows resuming training from checkpoints and loading pre-trained weights.

- **Performance Tracking**:
  - Includes logging to track model performance, loss metrics, and training progress.
  - Automatically saves model checkpoints during training.

## Performance Metrics

- **Dice Coefficient**: Measures the overlap between predicted masks and ground truth.
- **Intersection over Union (IoU)**: Evaluates the accuracy of predictions relative to the size of the union of predicted and ground truth regions.

## Data Folder

The `Data` folder contains utilities for data loading and processing. The `BraTSDataset` class handles data loading, augmentation, and preprocessing. Key highlights:

### BraTS 2024 Data Loading

- **Batch Data Loading**: Efficiently loads BraTS 2024 dataset from the disk in batches for both training and validation.
- **Normalization and Resizing**: Automatically resizes images and normalizes data for model input.
- **Mask Preprocessing**: Handles preprocessing of segmentation masks for training.
- **Modality Handling**: Works with various data modalities (`t1`, `t2`, `flair`, etc.) and stacks them as input channels.

```python
class BraTSDataset(Dataset):
    """Dataset class for loading BraTS 2024 dataset for tumour segmentation."""
    
    def __getitem__(self, index):
        # Load and preprocess image and mask
        ...
        
    def load_img(self, file_path: str):
        """Load .nii.gz images from file."""
        ...
        
    def normalize(self, data: np.ndarray):
        """Normalize image data."""
        ...
        
    def resize(self, data: np.ndarray):
        """Resize image data."""
        ...
        
    def preprocess_mask_labels(self, mask: np.ndarray):
        """Process tumour masks into multi-channel masks for training."""
        ...
