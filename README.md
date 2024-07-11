# F2NetViT

python train.py

This script will:

Load the configuration from config/config.yaml.

Create the datasets and dataloaders using data_provider_brats.py.

Apply the necessary data transformations using transforms.py.

Train the 3D U-Net model defined in unet3d.py.

Save model checkpoints during training.