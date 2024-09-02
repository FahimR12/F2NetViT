from tensorflow.keras.utils import Sequence
import numpy as np
import os
import json
import settings
import nibabel as nib
import numpy as np
    

def get_decathlon_filelist(data_path, seed=816, split=0.80):
    """
    Get the paths for the files from bin_dataset.json and split into training and validation.
    """
    data_path = '/home/fahim/nnUNet/nnUNet_raw/Dataset111_BraTS2024'
    json_filename = os.path.join(data_path, "bin_dataset.json")

    try:
        with open(json_filename, "r") as fp:
            experiment_data = json.load(fp)
    except IOError as e:
        raise Exception("File {} doesn't exist. It should be part of the "
                        "Decathlon directory".format(json_filename))

    # Print information about the dataset
    print("*" * 30)
    print("=" * 30)
    print("Dataset name:        ", experiment_data["name"])
    print("Dataset description: ", experiment_data["description"])
    print("Tensor image size:   ", experiment_data["tensorImageSize"])
    print("Dataset release:     ", experiment_data["release"])
    print("Dataset reference:   ", experiment_data["reference"])
    print("Dataset license:     ", experiment_data["licence"])
    print("=" * 30)
    print("*" * 30)

    # Set the random seed for reproducibility
    np.random.seed(seed)
    numFiles = experiment_data["numTraining"]
    idxList = np.arange(numFiles)  # List of file indices
    np.random.shuffle(idxList)  # Shuffle the indices to randomize train/test/split

    trainIdx = int(np.floor(numFiles * split))  # Index for the end of the training files
    trainList = idxList[:trainIdx]
    validateList = idxList[trainIdx:]

    trainFiles = []
    for idx in trainList:
        trainFiles.append(os.path.join(data_path, experiment_data["training"][idx]["label"]))

    validateFiles = []
    for idx in validateList:
        validateFiles.append(os.path.join(data_path, experiment_data["training"][idx]["label"]))

    print("Number of training files   = {}".format(len(trainList)))
    print("Number of validation files = {}".format(len(validateList)))

    # Return train and validation file lists
    return trainFiles, validateFiles


class DatasetGenerator(Sequence):
    """
    TensorFlow Dataset from Python/NumPy Iterator
    """
    
    def __init__(self, filenames, batch_size=8, crop_dim=[240, 240], augment=False, seed=816):
        self.filenames = filenames
        self.batch_size = batch_size
        self.crop_dim = crop_dim
        self.augment = augment
        self.seed = seed
        
        # Load the first image to determine the number of slices per scan
        img = np.array(nib.load(filenames[0]).dataobj)
        print(f"Initial image shape: {img.shape}, dtype: {img.dtype}")  # Debugging print

        self.slice_dim = 2  # We'll assume z-dimension (slice) is last
        self.num_slices_per_scan = img.shape[self.slice_dim]  # Determine the number of slices

        if crop_dim[0] == -1:
            self.crop_dim[0] = img.shape[0]
        if crop_dim[1] == -1:
            self.crop_dim[1] = img.shape[1]

        print(f"Crop dimensions set to: {self.crop_dim}")  # Debugging print

        self.num_files = len(self.filenames)
        self.ds = self.get_dataset()

    def preprocess_img(self, img):
        """
        Preprocessing for the image: z-score normalize
        """
        print(f"Preprocessing image with original shape: {img.shape}, dtype: {img.dtype}")  # Debugging print
        img = (img - img.mean()) / img.std()
        print(f"Image after preprocessing: mean={img.mean()}, std={img.std()}")  # Debugging print
        return img

    def preprocess_label(self, label):
        """
        Convert all labels to binary (WT vs. background) for binary segmentation.
        """
        print(f"Preprocessing label with original unique values: {np.unique(label)}")  # Debugging print
        label[label > 0] = 1.0
        print(f"Label after preprocessing unique values: {np.unique(label)}")  # Debugging print
        return label

        
    def augment_data(self, img, msk):
        """
        Data augmentation: Flip image and mask. Rotate image and mask.
        """
        print(f"Augmenting data with original shape: img={img.shape}, msk={msk.shape}")  # Debugging print
        if np.random.rand() > 0.5:
            ax = np.random.choice([0, 1])
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        if np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            img = np.rot90(img, rot, axes=[0, 1])  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=[0, 1])  # Rotate axes 0 and 1

        print(f"Data after augmentation: img shape={img.shape}, msk shape={msk.shape}")  # Debugging print
        return img, msk


    def crop_input(self, img, msk):
        """
        Randomly crop the image and mask
        """
        print(f"Cropping input with original shape: img={img.shape}, msk={msk.shape}")  # Debugging print
        slices = []
        is_random = self.augment and np.random.rand() > 0.5  # Do we randomize?

        for idx in range(2):  # Go through each dimension
            cropLen = self.crop_dim[idx]
            imgLen = img.shape[idx]
            start = (imgLen - cropLen) // 2
            ratio_crop = 0.20  # Crop up to this % of pixels for offset
            offset = int(np.floor(start * ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    if (start + cropLen) > imgLen:  # Don't fall off the image
                        start = (imgLen - cropLen) // 2
            else:
                start = 0

            slices.append(slice(start, start + cropLen))

        img_cropped, msk_cropped = img[tuple(slices)], msk[tuple(slices)]
        print(f"Image and mask after cropping: img shape={img_cropped.shape}, msk shape={msk_cropped.shape}")  # Debugging print
        return img_cropped, msk_cropped


    def generate_batch_from_files(self):
        """
        Python generator which goes through a list of filenames to load.
        The files are 3D images (slice is dimension index 2 by default). 
        We yield them as a batch of 2D slices.
        """
        np.random.seed(self.seed)  # Set a random seed
        idx = 0
        idy = 0

        while True:
            NUM_QUEUED_IMAGES = 1 + self.batch_size // self.num_slices_per_scan  # Get enough for full batch + 1

            for idz in range(NUM_QUEUED_IMAGES):
                label_filename = self.filenames[idx]

                # Correctly retrieve the corresponding image filename
                img_filename = label_filename.replace("labelsTr", "imagesTr").replace("binary_", "").replace(".nii.gz", "_0003.nii.gz")

                # Debug print statements to verify paths
                print(f"Attempting to load image from: {img_filename}")
                print(f"Attempting to load label from: {label_filename}")

                try:
                    # Load the image and label
                    img = np.array(nib.load(img_filename).dataobj)
                    print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")  # Debugging print
                    
                    # Check if the image has a channel dimension
                    if img.ndim == 4:
                        img = img[:, :, :, 0]  # Just take FLAIR channel (channel 0)

                    img = self.preprocess_img(img)

                    label = np.array(nib.load(label_filename).dataobj)
                    print(f"Loaded label shape: {label.shape}, dtype: {label.dtype}")  # Debugging print
                    label = self.preprocess_label(label)

                except FileNotFoundError as e:
                    print(f"Error loading file: {e}")
                    continue  # or handle as appropriate

                # Crop input and label
                img, label = self.crop_input(img, label)
                print(f"Image and label shapes after cropping: img shape={img.shape}, label shape={label.shape}")  # Debugging print

                # Continue with the rest of the method...


                if idz == 0:
                    img_stack = img
                    label_stack = label
                else:
                    img_stack = np.concatenate((img_stack, img), axis=self.slice_dim)
                    label_stack = np.concatenate((label_stack, label), axis=self.slice_dim)
                
                idx += 1 
                if idx >= len(self.filenames):
                    idx = 0
                    np.random.shuffle(self.filenames)  # Shuffle the filenames for the next iteration
            
            img = img_stack
            label = label_stack
            num_slices = img.shape[self.slice_dim]
            
            if self.batch_size > num_slices:
                raise Exception(f"Batch size {self.batch_size} is greater than"
                                f" the number of slices in the image {num_slices}."
                                " Data loader cannot be used.")

            if self.augment:
                slice_idx = np.random.choice(range(num_slices), num_slices)
                img = img[:, :, slice_idx]  # Randomize the slices
                label = label[:, :, slice_idx]

            if (idy + self.batch_size) < num_slices:  # We have enough slices for batch
                img_batch = img[:, :, idy:idy + self.batch_size]
                label_batch = label[:, :, idy:idy + self.batch_size]   
            else:  # We need to pad the batch with slices
                img_batch = img[:, :, -self.batch_size:]
                label_batch = label[:, :, -self.batch_size:]

            if self.augment:
                img_batch, label_batch = self.augment_data(img_batch, label_batch)
                
            if len(np.shape(img_batch)) == 3:
                img_batch = np.expand_dims(img_batch, axis=-1)
            if len(np.shape(label_batch)) == 3:
                label_batch = np.expand_dims(label_batch, axis=-1)
                
            yield np.transpose(img_batch, [2, 0, 1, 3]).astype(np.float32), np.transpose(label_batch, [2, 0, 1, 3]).astype(np.float32)

            idy += self.batch_size
            if idy >= num_slices:  # We finished this file, move to the next
                idy = 0
                idx += 1

            if idx >= len(self.filenames):
                idx = 0
                np.random.shuffle(self.filenames)  # Shuffle the filenames for the next iteration
    def get_input_shape(self):
        """
        Get image shape
        """
        return [self.crop_dim[0], self.crop_dim[1], 1]
        
    def get_output_shape(self):
        """
        Get label shape
        """
        return [self.crop_dim[0], self.crop_dim[1], 1] 
    
    def get_dataset(self):
        """
        Return a dataset
        """
        ds = self.generate_batch_from_files()
        return ds  
    
    def __len__(self):
        return (self.num_slices_per_scan * self.num_files) // self.batch_size

    def __getitem__(self, idx):
        return next(self.ds)
        
    def plot_samples(self):
        """
        Plot some random samples
        """
        import matplotlib.pyplot as plt
        
        img, label = next(self.ds)
        
        print(img.shape)
 
        plt.figure(figsize=(10, 10))
        
        slice_num = 3
        plt.subplot(2, 2, 1)
        plt.imshow(img[slice_num, :, :, 0])
        plt.title(f"MRI, Slice #{slice_num}")

        plt.subplot(2, 2, 2)
        plt.imshow(label[slice_num, :, :, 0])
        plt.title(f"Tumor, Slice #{slice_num}")

        slice_num = self.batch_size - 1
        plt.subplot(2, 2, 3)
        plt.imshow(img[slice_num, :, :, 0])
        plt.title(f"MRI, Slice #{slice_num}")

        plt.subplot(2, 2, 4)
        plt.imshow(label[slice_num, :, :, 0])
        plt.title(f"Tumor, Slice #{slice_num}")
