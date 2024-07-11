import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom
from config.reader import read_config

class data_loader_3D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', modality=4, multimodal=True, site=None, image_downsample=True):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test

        modality: int
            Modality of the MR sequence
            1: T1
            2: T1Gd
            3: T2
            4: T2-FLAIR

        site: str
            Name of the client for federated learning

        image_downsample: bool
            If we want to have image down sampling
        """

        # Read configuration file
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.mode = mode

        # Determine the base directory based on the mode (train/valid)
        if mode == 'train':
            self.file_base_dir = self.params['T_DATASET_PATH']
            self.dataset_base_path = self.params['TRAIN_BASE_PATH']
        else:
            self.file_base_dir = self.params['V_DATASET_PATH']
            self.dataset_base_path = self.params['VALID_BASE_PATH']

        # Read the file paths from the file list
        self.file_path_list = self._get_file_paths(self.file_base_dir)
        
        # Store other parameters
        self.modality = int(modality)
        self.multimodal = multimodal
        self.image_downsample = image_downsample

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor (if mode is 'train')
        """
        # Construct the full paths for patient directory and label file
        sub_dir = self.file_path_list[idx]
        path_pat = os.path.join(self.dataset_base_path, sub_dir)
        label_path = os.path.join(path_pat, sub_dir + '-seg.nii.gz')

        # Debugging print statements
        print(f"Path to patient directory: {path_pat}")
        if self.mode == 'train':
            print(f"Path to label: {label_path}")

        # Load and preprocess the label if in training mode
        if self.mode == 'train':
            label = nib.load(label_path).get_fdata()  # (h, w, d)
            label = label.transpose(2, 0, 1)  # (d, h, w)
            label = label.astype(np.int32)  # (d, h, w)

            label1 = label.copy()  # (d, h, w)
            label2 = label.copy()  # (d, h, w)
            label4 = label.copy()  # (d, h, w)
            label1 = np.where(label1 == 1, 1, 0)  # (d, h, w)
            label2 = np.where(label2 == 2, 1, 0)  # (d, h, w)
            label4 = np.where(label4 == 4, 1, 0)  # (d, h, w)

        # Process multimodal images if required
        if self.multimodal:
            path_file1 = os.path.join(path_pat, sub_dir + '-t1n.nii.gz')
            img1 = nib.load(path_file1).get_fdata()  # (h, w, d)
            normalized_img1 = self.irm_min_max_preprocess(img1)  # (h, w, d)
            normalized_img1 = normalized_img1.transpose(2, 0, 1)  # (d, h, w)

            path_file2 = os.path.join(path_pat, sub_dir + '-t1c.nii.gz')
            img2 = nib.load(path_file2).get_fdata()  # (h, w, d)
            normalized_img2 = self.irm_min_max_preprocess(img2)  # (h, w, d)
            normalized_img2 = normalized_img2.transpose(2, 0, 1)  # (d, h, w)

            path_file3 = os.path.join(path_pat, sub_dir + '-t2w.nii.gz')
            img3 = nib.load(path_file3).get_fdata()  # (h, w, d)
            normalized_img3 = self.irm_min_max_preprocess(img3)  # (h, w, d)
            normalized_img3 = normalized_img3.transpose(2, 0, 1)  # (d, h, w)

            path_file4 = os.path.join(path_pat, sub_dir + '-t2f.nii.gz')
            img4 = nib.load(path_file4).get_fdata()  # (h, w, d)
            normalized_img4 = self.irm_min_max_preprocess(img4)  # (h, w, d)
            normalized_img4 = normalized_img4.transpose(2, 0, 1)  # (d, h, w)

            # Image resizing for memory issues
            if self.image_downsample:
                if self.mode == 'train':
                    normalized_img_resized1, label1 = self.resize_manual(normalized_img1, label1)
                    normalized_img_resized2, label2 = self.resize_manual(normalized_img2, label2)
                    normalized_img_resized3, label4 = self.resize_manual(normalized_img3, label4)
                    normalized_img_resized4, _ = self.resize_manual(normalized_img4, label4)
                else:
                    normalized_img_resized1 = self.resize_manual(normalized_img1, None)
                    normalized_img_resized2 = self.resize_manual(normalized_img2, None)
                    normalized_img_resized3 = self.resize_manual(normalized_img3, None)
                    normalized_img_resized4 = self.resize_manual(normalized_img4, None)
                normalized_img_resized = np.stack((normalized_img_resized1, normalized_img_resized2,
                                                   normalized_img_resized3, normalized_img_resized4))  # (c=4, d, h, w)
            else:
                normalized_img_resized = np.stack((normalized_img1, normalized_img2,
                                                   normalized_img3, normalized_img4))  # (c=4, d, h, w)
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (c=4, d, h, w)

        else:
            if self.modality == 1:
                path_file = os.path.join(path_pat, sub_dir + '-t1n.nii.gz')
            elif self.modality == 2:
                path_file = os.path.join(path_pat, sub_dir + '-t1c.nii.gz')
            elif self.modality == 3:
                path_file = os.path.join(path_pat, sub_dir + '-t2w.nii.gz')
            elif self.modality == 4:
                path_file = os.path.join(path_pat, sub_dir + '-t2f.nii.gz')
            img = nib.load(path_file).get_fdata()
            img = img.astype(np.float32)  # (h, w, d)
            normalized_img = self.irm_min_max_preprocess(img)  # (h, w, d)
            normalized_img = normalized_img.transpose(2, 0, 1)  # (d, h, w)

            # Image resizing for memory issues
            if self.image_downsample:
                normalized_img_resized = self.resize_manual(normalized_img, None)
            else:
                normalized_img_resized = normalized_img
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (d, h, w)
            normalized_img_resized = torch.unsqueeze(normalized_img_resized, 0)  # (c=1, d, h, w)

        if self.mode == 'train':
            label1 = torch.from_numpy(label1)  # (d, h, w)
            label2 = torch.from_numpy(label2)  # (d, h, w)
            label4 = torch.from_numpy(label4)  # (d, h, w)
            label = torch.stack((label1, label2, label4))  # (c=3, d, h, w)

            normalized_img_resized = normalized_img_resized.float()  # float32
            label = label.long()  # int64 (long)

            # Ensuring the output sizes match
            target_size = tuple(self.params['Network']['resize_shape'])
            normalized_img_resized, label = self.resize_manual(normalized_img_resized, label)
            
            return normalized_img_resized, label

        else:
            normalized_img_resized = normalized_img_resized.float()  # float32
            return normalized_img_resized

    def data_normalization_mean_std(self, image):
        """Subtracting mean and std for each individual patient and modality
        mean and std only over the tumor region

        Parameters
        ----------
        image: numpy array
            The raw input image
        Returns
        -------
        normalized_img: numpy array
            The normalized image
        """
        mean = image[image > 0].mean()
        std = image[image > 0].std()

        if self.outzero_normalization:
            image[image < 0] = -1000

        normalized_img = (image - mean) / std

        if self.outzero_normalization:
            normalized_img[normalized_img < -100] = 0

        return normalized_img

    def irm_min_max_preprocess(self, image, low_perc=1, high_perc=99):
        """Main pre-processing function used for the challenge (seems to work the best).
        Remove outliers voxels first, then min-max scale.
        Warnings
        --------
        This will not do it channel wise!!
        """
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)

        min_ = np.min(image)
        max_ = np.max(image)
        scale = max_ - min_
        image = (image - min_) / scale

        return image

    def resize_manual(self, img, gt):
        """Downsampling of the image and its label.
        Parameters
        ----------
        img: numpy array
            Input image
        gt: numpy array
            Input label
        Returns
        -------
        img: numpy array
            Downsampled image
        gt: numpy array
            Downsampled label
        """
        resize_ratio = np.divide(tuple(self.params['Network']['resize_shape']), img.shape[1:])
        img = zoom(img, (1, *resize_ratio), order=2)
        if gt is not None:
            gt = zoom(gt, (1, *resize_ratio), order=0)
        return img, gt

    def _get_file_paths(self, file_base_dir):
        """Utility function to get the file paths from the given directory.
        Parameters
        ----------
        file_base_dir: str
            Base directory containing the files.
        Returns
        -------
        file_paths: list
            List of file paths.
        """
        with open(file_base_dir, 'r') as f:
            file_paths = f.read().splitlines()
        return file_paths
