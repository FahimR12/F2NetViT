import nibabel as nib
import numpy as np

def load_nifti_image(file_path):
    """
    Load a NIfTI image and return its data as a numpy array.
    """
    img = nib.load(file_path)
    return img.get_fdata()

def create_class_mask(segmentation, class_value):
    """
    Create a binary mask for a specific class value from the segmentation data.

    :param segmentation: Numpy array of the segmentation data.
    :param class_value: The class value to create a mask for.
    :return: Binary mask for the class.
    """
    return (segmentation == class_value).astype(np.int32)

def calculate_dice_score(pred_mask, gt_mask):
    """
    Calculate the Dice coefficient between two binary masks.

    :param pred_mask: Predicted binary mask.
    :param gt_mask: Ground truth binary mask.
    :return: Dice coefficient as a float.
    """
    intersection = np.sum(pred_mask * gt_mask)
    sum_masks = np.sum(pred_mask) + np.sum(gt_mask)
    
    if sum_masks == 0:
        return 1.0  # To handle cases where both prediction and ground truth are empty
    else:
        return 2.0 * intersection / sum_masks

def remap_labels(data, mapping):
    """
    Remap the labels in the data array according to the given mapping.
    :param data: numpy array with the original labels.
    :param mapping: dictionary where keys are original labels and values are the desired labels.
    :return: remapped numpy array.
    """
    remapped_data = np.copy(data)
    for orig_label, new_label in mapping.items():
        remapped_data[data == orig_label] = new_label
    return remapped_data

def main(prediction_file, ground_truth_file, class_value, label_mapping=None):
    """
    Main function to load prediction and ground truth files, create class masks, and calculate the Dice score.

    :param prediction_file: Path to the NIfTI file containing the predicted segmentation.
    :param ground_truth_file: Path to the NIfTI file containing the ground truth segmentation.
    :param class_value: The class value to calculate the Dice score for.
    :param label_mapping: Optional dictionary to remap predicted labels to match ground truth labels.
    """
    # Load the predicted and ground truth segmentations
    prediction = load_nifti_image(prediction_file)
    ground_truth = load_nifti_image(ground_truth_file)
    
    # Remap the predicted labels if a mapping is provided
    if label_mapping:
        prediction = remap_labels(prediction, label_mapping)
    
    # Create masks for the class of interest
    prediction_mask = create_class_mask(prediction, class_value)
    ground_truth_mask = create_class_mask(ground_truth, class_value)
    
    # Calculate the Dice score
    dice_score = calculate_dice_score(prediction_mask, ground_truth_mask)
    
    print(f"Dice Score for Class {class_value}: {dice_score:.4f}")

if __name__ == "__main__":
    # Replace these with your actual file paths
    prediction_file = "/home/fahim/nnUNet/nnUNet_results/Dataset101_BraTS2024/nnUNetTrainer__nnUNetPlans__2d_v1/fold_2/validation/BraTS_GLI_00485_100.nii.gz"
    ground_truth_file =  "/home/fahim/nnUNet/nnUNet_preprocessed/Dataset101_BraTS2024/gt_segmentations/BraTS_GLI_00485_100.nii.gz"

    # Define the correct mapping
    label_mapping = {
        4: 3,  # Predicted label 4 should be 1 in ground truth
        3: 4,  # Predicted label 3 should be 2 in ground truth
        2: 1,  # Predicted label 2 should be 3 in ground truth
        1: 2   # Predicted label 1 should be 4 in ground truth
    }

    # Run for each class
    for class_value in [1, 2, 3, 4]:
        main(prediction_file, ground_truth_file, class_value, label_mapping)
