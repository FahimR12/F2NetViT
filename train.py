import math
import os
import torch
from config.reader import read_config
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Data.data_provider_brats import data_loader_3D
from unet3d import UNet3D
from transforms import train_transform, train_transform_cuda, val_transform, val_transform_cuda

# Configuration file path
cfg_path = r'C:\Users\fahim\Documents\Research_project\F2NetViT\config\config.yaml'
config = read_config(cfg_path)

# Create datasets and dataloaders
train_dataset = data_loader_3D(cfg_path, mode='train', modality=4, multimodal=True, image_downsample=config['Network']['image_downsample'])
val_dataset = data_loader_3D(cfg_path, mode='valid', modality=4, multimodal=True, image_downsample=config['Network']['image_downsample'])

train_loader = DataLoader(train_dataset, batch_size=config['Network']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['Network']['batch_size'], shuffle=False)

writer = SummaryWriter("runs")

model = UNet3D(in_channels=4, num_classes=3)  # Assuming 4 input channels (modalities) and 3 output classes (labels)

# Check if GPU is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() and config.get('TRAIN_CUDA', False) else "cpu")
model = model.to(device)
print(f"Using device: {device}")

criterion = CrossEntropyLoss(weight=torch.Tensor(config['class_weights']).to(device) if 'class_weights' in config else None)
optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

# Ensure the checkpoints directory exists
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Set iteration limit for debugging
iteration_limit = 5

for epoch in range(config['num_epochs']):
    train_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        if i >= iteration_limit:
            break
        image, ground_truth = data
        image, ground_truth = image.to(device), ground_truth.to(device)

        optimizer.zero_grad()
        target = model(image.float())

        # Reduce ground_truth to the correct shape
        ground_truth = torch.argmax(ground_truth, dim=1)

        # Ensure ground_truth is long (int64)
        loss = criterion(target, ground_truth.long())
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= iteration_limit:
                break
            image = data.to(device)  # Only image data for validation

            target = model(image.float())

            # Debugging print statements
            print(f"target shape: {target.shape}")
            print(f"target dtype: {target.dtype}")
            print(f"target device: {target.device}")

    writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased ({min_valid_loss:.6f} --> {valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'epoch{epoch+1}_valLoss{min_valid_loss:.6f}.pth'))

writer.flush()
writer.close()