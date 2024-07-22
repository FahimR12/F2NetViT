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
    def __init__(self, cfg_path, mode='train', modality=4, multimodal=True, site=None, image_downsample=True, transform=None):
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

        transform: callable
            A function/transform to apply to the data
        """

        # Read configuration file
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.mode = mode

        # Determine the base directory based on the mode (train/valid)
        if mode == 'train':
            # hold the train_list.txt
            self.file_base_dir = self.params['T_DATASET_PATH']

            # holds the path to the training dataset
            self.dataset_base_path = self.params['TRAIN_BASE_PATH']
        else:
            self.file_base_dir = self.params['V_DATASET_PATH']
            self.dataset_base_path = self.params['VALID_BASE_PATH']

        # - Read the file paths from the file list
        # now have the train\valid_list.txt as an actual list to use for the for loop.
        self.file_path_list = self._get_file_paths(self.file_base_dir)
        
        # Store other parameters
        self.modality = int(modality)
        self.multimodal = multimodal
        self.image_downsample = image_downsample
        self.transform = transform

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
        label: torch tensor (or None if in validation mode without labels)
        """
       
        #index the list of files
        sub_dir = self.file_path_list[idx]

        # - Construct the full paths for patient directory and label file (Segmentation mask .gz files)
        path_pat = os.path.join(self.dataset_base_path, sub_dir)
        label_path = os.path.join(path_pat, sub_dir + '-seg.nii.gz')

        # Debugging print statements
        print(f"Path to patient directory: {path_pat}")
        print(f"Path to label: {label_path}")

        # Initialize label as None
        label = None
        
        # Load and preprocess the label if it exists
        if os.path.exists(label_path):
            label = nib.load(label_path).get_fdata()  # (h, w, d)
            label = label.transpose(2, 0, 1)  # (d, h, w)
            label = label.astype(np.int32)  # (d, h, w)

            # - Should have 4 classes for labels: Non-enhancing tumor core (NETC — label 1),
            #                                     Surrounding non-enhancing FLAIR hyperintensity (SNFH) — label 2),
            #                                     Enhancing tissue (ET — label 3) and,
            #                                     Resection cavity (RC - label 4)
            label1 = label.copy()  # (d, h, w)
            label2 = label.copy()  # (d, h, w)
            label3 = label.copy()  # (d, h, w)
            label4 = label.copy()  # (d, h, w)
            label1 = np.where(label1 == 1, 1, 0)  # (d, h, w)
            label2 = np.where(label2 == 2, 1, 0)  # (d, h, w)
            label3 = np.where(label3 == 3, 1, 0)  # (d, h, w)
            label4 = np.where(label4 == 4, 1, 0)  # (d, h, w)
            
            label1 = torch.from_numpy(label1)  # (d, h, w)
            label2 = torch.from_numpy(label2)  # (d, h, w)
            label3 = torch.from_numpy(label3)  # (d, h, w)
            label4 = torch.from_numpy(label4)  # (d, h, w)
            label = torch.stack((label1, label2, label3, label4))  # (c=4, d, h, w)
            label = label.long()  # int64 (long)

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
                normalized_img_resized1 = self.resize_manual(normalized_img1)
                normalized_img_resized2 = self.resize_manual(normalized_img2)
                normalized_img_resized3 = self.resize_manual(normalized_img3)
                normalized_img_resized4 = self.resize_manual(normalized_img4)
                normalized_img_resized = np.stack((normalized_img_resized1, normalized_img_resized2,
                                                   normalized_img_resized3, normalized_img_resized4))
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
                normalized_img_resized = self.resize_manual(normalized_img)
            else:
                normalized_img_resized = normalized_img
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (d, h, w)
            normalized_img_resized = torch.unsqueeze(normalized_img_resized, 0)  # (c=1, d, h, w)

        normalized_img_resized = normalized_img_resized.float()  # float32

        if label is not None:
            normalized_img_resized, label = self.resize_manual(normalized_img_resized, label)
        
        print(f"(data_provider_brats.py) Before transform - Image shape: {normalized_img_resized.shape}, Label shape: {label.shape if label is not None else 'None'}")

        # Apply transformations if provided
        if self.transform:
            sample = {"image": normalized_img_resized}
            if label is not None:
                sample["label"] = label
            sample = self.transform(sample)
            normalized_img_resized = sample["image"]
            if "label" in sample:
                label = sample["label"]

        print(f"(data_provider_brats.py) After transform - Image shape: {normalized_img_resized.shape}, Label shape: {label.shape if label is not None else 'None'}")
        # Ensure neither normalized_img_resized nor label is None
        if normalized_img_resized is None or (self.mode == 'train' and label is None):
            raise ValueError("normalized_img_resized or label is None before returning.")

        return normalized_img_resized, label
        
    

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

    def resize_manual(self, img, gt=None):
        """Downsampling of the image and its label.
        Parameters
        ----------
        img: numpy array
            Input image
        gt: numpy array
            Input label (optional)
        Returns
        -------
        img: numpy array
            Downsampled image
        gt: numpy array
            Downsampled label (if provided)
        """
        resize_ratio = np.divide(tuple(self.params['Network']['resize_shape']), img.shape[1:])
        img = zoom(img, (1, *resize_ratio), order=2)
        if gt is not None:
            gt = zoom(gt, (1, *resize_ratio), order=0)
        return img if gt is None else (img, gt)

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
