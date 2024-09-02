import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def lesion_wise_dice_loss(pred, target, smooth=1e-5):
    """
    Calculate lesion-wise Dice loss.
    This assumes binary classification and expects the target to be a one-hot encoded tensor.
    """
    pred = torch.sigmoid(pred)
    pred = pred > 0.5  # Convert probabilities to binary

    intersection = (pred * target).sum(dim=[2, 3, 4])
    union = pred.sum(dim=[2, 3, 4]) + target.sum(dim=[2, 3, 4])

    lesion_dice = (2. * intersection + smooth) / (union + smooth)
    lesion_dice_loss = 1 - lesion_dice.mean()
    return lesion_dice_loss


def calculate_lesion_wise_dice(self, predicted_segmentation, target):
    """
    Calculate lesion-wise Dice coefficient.

    Args:
        predicted_segmentation (torch.Tensor): The predicted segmentation (one-hot encoded).
        target (torch.Tensor): The ground truth segmentation (one-hot encoded).

    Returns:
        float: The lesion-wise Dice coefficient.
    """
    eps = 1e-6  # Small constant to avoid division by zero

    # Flatten the segmentation maps to calculate individual lesion-wise metrics
    predicted_flat = predicted_segmentation.view(predicted_segmentation.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Calculate intersection and union for each lesion
    intersection = (predicted_flat * target_flat).sum(1)
    union = predicted_flat.sum(1) + target_flat.sum(1)

    dice = (2.0 * intersection + eps) / (union + eps)

    return dice.mean().item()

def cal_hd95(pred, target):
    """
    Compute the 95th percentile Hausdorff Distance.
    :param pred: Predicted segmentation mask.
    :param target: Ground truth segmentation mask.
    :return: HD95 distance.
    """
    pred = pred > 0.5
    target = target > 0.5

    # Convert to numpy
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # Find surface points
    pred_surface = np.argwhere(pred_np)
    target_surface = np.argwhere(target_np)

    # Compute directed Hausdorff distances
    forward_hd = directed_hausdorff(pred_surface, target_surface)[0]
    backward_hd = directed_hausdorff(target_surface, pred_surface)[0]

    # Take the maximum of the two directed distances
    hd95 = max(forward_hd, backward_hd)
    
    return hd95

