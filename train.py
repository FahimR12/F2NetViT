import math
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

writer = SummaryWriter("runs")

model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES)
train_transforms = train_transform
val_transforms = val_transform

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    model.train()
    for data in train_dataloader:
        image, ground_truth = data['image'], data['label']
        optimizer.zero_grad()
        target = model(image)
        loss = criterion(target, ground_truth)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    valid_loss = 0.0
    model.eval()
    for data in val_dataloader:
        image, ground_truth = data['image'], data['label']
        
        target = model(image)
        loss = criterion(target,ground_truth)
        valid_loss = loss.item()
        
    writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)
    
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
    
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

writer.flush()
writer.close()

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from config.reader import read_config
from transforms import train_transform, val_transform, train_transform_cuda, val_transform_cuda

class data_loader_3D(Dataset):
    def __init__(self, cfg_path, mode='train', modality=4, multimodal=True, site=None, image_downsample=True):
        
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.mode = mode

        if mode == 'train':
            self.file_base_dir = self.params['T_DATASET_PATH']
            self.dataset_base_path = self.params['TRAIN_BASE_PATH']
        else:
            self.file_base_dir = self.params['V_DATASET_PATH']
            self.dataset_base_path = self.params['VALID_BASE_PATH']

        self.file_path_list = self._get_file_paths(self.file_base_dir)
        self.modality = int(modality)
        self.multimodal = multimodal
        self.image_downsample = image_downsample

        # Select transforms based on mode and device
        self.transform = train_transform if mode == 'train' else val_transform
        if torch.cuda.is_available() and self.params.get('TRAIN_CUDA', False):
            self.transform = train_transform_cuda if mode == 'train' else val_transform_cuda

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        sub_dir = self.file_path_list[idx]
        path_pat = os.path.join(self.dataset_base_path, sub_dir)
        label_path = os.path.join(path_pat, sub_dir + '-seg.nii.gz')

        label = nib.load(label_path).get_fdata()
        label = label.transpose(2, 0, 1)
        label = label.astype(np.int32)

        label1 = label.copy()
        label2 = label.copy()
        label4 = label.copy()
        label1 = np.where(label1 == 1, 1, 0)
        label2 = np.where(label2 == 2, 1, 0)
        label4 = np.where(label4 == 4, 1, 0)

        if self.multimodal:
            img_files = ['-t1n.nii.gz', '-t1c.nii.gz', '-t2w.nii.gz', '-t2f.nii.gz']
            images = []
            for img_file in img_files:
                img_path = os.path.join(path_pat, sub_dir + img_file)
                img = nib.load(img_path).get_fdata()
                img = self.irm_min_max_preprocess(img)
                img = img.transpose(2, 0, 1)
                images.append(img)
            normalized_img_resized = np.stack(images)
        else:
            img_file = ['-t1n.nii.gz', '-t1c.nii.gz', '-t2w.nii.gz', '-t2f.nii.gz'][self.modality - 1]
            img_path = os.path.join(path_pat, sub_dir + img_file)
            img = nib.load(img_path).get_fdata()
            img = img.astype(np.float32)
            img = self.irm_min_max_preprocess(img)
            img = img.transpose(2, 0, 1)
            normalized_img_resized = np.expand_dims(img, axis=0)

        if self.image_downsample:
            normalized_img_resized, label1 = self.resize_manual(normalized_img_resized, label1)
            _, label2 = self.resize_manual(normalized_img_resized, label2)
            _, label4 = self.resize_manual(normalized_img_resized, label4)

        label1 = torch.from_numpy(label1)
        label2 = torch.from_numpy(label2)
        label4 = torch.from_numpy(label4)
        label = torch.stack((label1, label2, label4))

        data_dict = {'image': torch.from_numpy(normalized_img_resized).float(), 'label': label.int()}

        # Apply transforms
        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def irm_min_max_preprocess(self, image, low_perc=1, high_perc=99):
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)
        min_, max_ = np.min(image), np.max(image)
        image = (image - min_) / (max_ - min_)
        return image

    def resize_manual(self, img, gt):
        resize_ratio = np.divide(tuple(self.params['Network']['resize_shape']), img.shape[1:])
        img = zoom(img, (1, *resize_ratio), order=2)
        gt = zoom(gt, resize_ratio, order=0)
        return img, gt

    def _get_file_paths(self, file_base_dir):
        with open(file_base_dir, 'r') as f:
            file_paths = f.read().splitlines()
        return file_paths