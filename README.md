# F2NetViT: A Research Repository for Advanced U-Net Architectures

This repository contains various U-Net based architectures developed for medical image segmentation tasks. The focus is on exploring and comparing the performance of multiple models, including Transformer-based and convolutional architectures. Below are the detailed descriptions of each model implemented in this repository.

## Models in This Repository

### 1. **F2NetViT**: A Transformer-Based Learning Network

The **F2NetViT** model is a Transformer-based architecture specifically designed for multi-modal data processing. The model's goal is to improve segmentation accuracy by leveraging its ability to extract multi-level features unique to each data modality.

**Challenges**:  
During implementation, several issues were encountered, particularly with data loading and validation. The most critical issue was a `TypeError`, which indicated that the batch must contain valid tensors, numpy arrays, numbers, dicts, or lists, but a `NoneType` was found. This error stemmed from discrepancies in how the validation dataset was handled, as the validation set lacked the masks used in the training set. Addressing these issues in future work will be crucial to ensure proper data handling and robust model performance.

---

### 2. **SWINUNetR Model**: Exploring Swin Transformer Blocks

The **SWINUNetR** model integrates **Swin Transformer** blocks to capture spatial hierarchies and long-range dependencies in medical images. This advanced architecture was designed to enhance the segmentation of complex tumor regions that are challenging for conventional U-Net models.

**Challenges**:  
Although the model was successfully built, significant computational limitations were encountered due to its high GPU memory requirements. Despite reducing the batch size and lowering image resolution, even a high-performance NVIDIA GeForce RTX 2080 Ti 12GB GPU could not run the model efficiently. This highlighted the need for greater memory resources when using such complex architectures.

---

### 3. **nnU-Net Modified**: Segmentation of Brain Tumors from the BraTS-GLI Dataset

The **nnU-Net** architecture, specifically its 2D and 3D variants, was employed for brain tumor segmentation tasks on the **BraTS-GLI dataset**. This model is capable of segmenting multiple tumor subregions, including NETC, ET, SNFH, and RC, in a computationally efficient manner.

#### 2D nnU-Net:
The 2D version of nnU-Net processes individual slices of 3D MRI volumes and utilizes an encoder-decoder structure. Skip connections are used to retain spatial information, and the architecture closely follows the original U-Net design. Modifications include replacing ReLU with leaky ReLU (negative slope of 0.01) and using instance normalization instead of batch normalization, as suggested by Isensee et al. (2021). This version prioritizes computational efficiency while maintaining strong segmentation performance.

---

### 4. **Intel 2D U-Net**: Optimized for Binary Segmentation

The **Intel 2D U-Net** is a binary segmentation model designed to distinguish tumor regions from the background. It serves as a benchmark to demonstrate U-Net's effectiveness in simpler segmentation tasks, making it particularly useful in clinical scenarios where fewer classes are involved.

**Architecture**:  
The Intel 2D U-Net follows the traditional U-Net design but has been optimized to run efficiently on Intel hardware. This model provides valuable insights into the U-Net architecture's performance in binary segmentation, emphasizing the need to select models based on the complexity of the segmentation challenge.

---

## Repository Structure

```plaintext
F2NetViT_model/      # F2NetViT model implementation
SWINUNetR_model/     # SWINUNetR model implementation
Intel_2D_UNet/       # Intel 2D U-Net implementation
nnUNet_modified/     # Modified nnU-Net implementation
.gitignore           # Git ignore file
README.md            # Project overview and details
