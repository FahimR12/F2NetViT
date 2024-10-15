# # F2NetViT: 3D U-Net for Brain Tumour Segmentation

This repository implements a 3D U-Net model for segmenting brain tumours using multi-modal MRI scans from the BRATS dataset.

## Quick Start

### Prerequisites

Ensure the following packages are installed:

- Python 3.x
- PyTorch
- TensorBoard
- NumPy
- PyYAML

python train.py

This script will:

Load the configuration from config/config.yaml.

Create the datasets and dataloaders using data_provider_brats.py.

Apply the necessary data transformations using transforms.py.

Train the 3D U-Net model defined in unet3d.py.

Save model checkpoints during training.


Install all dependencies:

```bash
pip install -r requirements.txt


