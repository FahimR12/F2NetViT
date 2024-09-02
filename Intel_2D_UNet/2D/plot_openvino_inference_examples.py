import os
import sys
import numpy as np
import time
import settings
import argparse
from dataloader import DatasetGenerator, get_decathlon_filelist
from openvino.inference_engine import IECore
import matplotlib
import matplotlib.pyplot as plt
import logging

# Use 'Agg' backend for matplotlib
matplotlib.use("Agg")

# Set up logging
logging.basicConfig(
    filename='inference_log.txt',  # Log file name
    level=logging.INFO,            # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Redirect stdout and stderr to log file
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ''
    
    def write(self, message):
        if message != '\n':  # Ignore newline-only messages
            self.level(message)
    
    def flush(self):
        self.buffer = ''

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

# Rest of your script remains unchanged

parser = argparse.ArgumentParser(
    description="OpenVINO Inference example for trained 2D U-Net model on BraTS.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="the path to the data")
parser.add_argument("--output_path", default=settings.OUT_PATH,
                    help="the folder to save the model and checkpoints")
parser.add_argument("--inference_filename", default=settings.INFERENCE_FILENAME,
                    help="the TensorFlow inference model filename")
parser.add_argument("--device", default="CPU",
                    help="the inference device")
parser.add_argument("--output_pngs", default="inference_examples",
                    help="the directory for the output prediction pngs")

parser.add_argument("--intraop_threads", default=settings.NUM_INTRA_THREADS,
                    type=int, help="Number of intra-op-parallelism threads")
parser.add_argument("--interop_threads", default=settings.NUM_INTER_THREADS,
                    type=int, help="Number of inter-op-parallelism threads")
parser.add_argument("--crop_dim", default=settings.CROP_DIM,
                    type=int, help="Crop dimension for images")
parser.add_argument("--seed", default=settings.SEED,
                    type=int, help="Random seed")
parser.add_argument("--split", type=float, default=settings.TRAIN_TEST_SPLIT,
                    help="Train/testing split for the data")

args = parser.parse_args()

# Your functions and main script remain unchanged

def calc_dice(target, prediction, smooth=0.0001):
    target = target.flatten()
    prediction = prediction.flatten()
    
    intersection = np.sum(target * prediction)
    dice = (2. * intersection + smooth) / (np.sum(target) + np.sum(prediction) + smooth)
    
    return dice

def calc_soft_dice(target, prediction, smooth=0.0001):
    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef

def calc_iou(target, prediction, smooth=0.0001):
    target = target.flatten()
    prediction = prediction.flatten()

    intersection = np.sum(target * prediction)
    union = np.sum(target) + np.sum(prediction) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou

def plot_results(ds, batch_num, png_directory, exec_net, input_layer_name, output_layer_name):
    plt.figure(figsize=(10, 10))
    img, msk = next(ds.ds)
    idx = np.argmax(np.sum(np.sum(msk[:, :, :, 0], axis=1), axis=1))

    plt.subplot(1, 3, 1)
    plt.imshow(img[idx, :, :, 0], cmap="bone", origin="lower")
    plt.title("MRI {}".format(idx), fontsize=20)

    plt.subplot(1, 3, 2)
    plt.imshow(msk[idx, :, :], cmap="bone", origin="lower")
    plt.title("Ground truth", fontsize=20)

    plt.subplot(1, 3, 3)

    logging.info("Index {}: ".format(idx))

    start_time = time.time()
    img_input = np.repeat(img[[idx]], 4, axis=-1)
    logging.info(f"Input shape: {img_input.shape}")  # Debugging print
    
    res = exec_net.infer({input_layer_name: img_input})
    prediction = np.squeeze(np.transpose(res[output_layer_name], [0, 2, 3, 1]), axis=2)

    logging.info("Output shape after squeeze: {}".format(prediction.shape))
    logging.info("Elapsed time = {:.4f} msecs".format(1000.0*(time.time()-start_time)))

    prediction = np.rot90(prediction, k=-1, axes=(1, 2))  
    prediction = np.flip(prediction, axis=2)              
    
    plt.imshow(prediction[0, :, :], cmap="bone", origin="lower")
    
    dice_coef = calc_dice(msk[idx], prediction[0])
    plt.title("Prediction\nDice = {:.4f}".format(dice_coef), fontsize=20)

    logging.info("Dice coefficient = {:.4f}".format(dice_coef))

    iou_coef = calc_iou(msk[idx], prediction[0])
    logging.info("IoU coefficient = {:.4f}".format(iou_coef))
    
    save_name = os.path.join(png_directory, "prediction_openvino_{}_{}.png".format(batch_num, idx))
    logging.info("Saved as: {}".format(save_name))
    plt.savefig(save_name)
    
    return dice_coef, iou_coef

if __name__ == "__main__":
    model_filename = os.path.join(args.output_path, args.inference_filename)

    trainFiles, validateFiles = get_decathlon_filelist(data_path=args.data_path, seed=args.seed, split=args.split)
    testFiles = []

    ds_test = DatasetGenerator(validateFiles, batch_size=128, crop_dim=[args.crop_dim,args.crop_dim], augment=False, seed=args.seed)
    
    if args.device != "CPU":
        precision = "FP16"
    else:
        precision = "FP32"
    path_to_xml_file = "{}.xml".format(os.path.join(args.output_path, precision, args.inference_filename))
    path_to_bin_file = "{}.bin".format(os.path.join(args.output_path, precision, args.inference_filename))

    ie = IECore()
    net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)

    input_layer_name = next(iter(net.input_info))
    output_layer_name = next(iter(net.outputs))
    logging.info("Input layer name = {}\nOutput layer name = {}".format(input_layer_name, output_layer_name))

    exec_net = ie.load_network(network=net, device_name=args.device, num_requests=1)

    png_directory = args.output_pngs
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    dice_scores = []
    iou_scores = []

    for batch_num in range(len(validateFiles)):
        try:
            dice, iou = plot_results(ds_test, batch_num, png_directory, exec_net, input_layer_name, output_layer_name)
            dice_scores.append(dice)
            iou_scores.append(iou)
        except StopIteration:
            break

    mean_dice = np.mean(dice_scores)
    sd_dice = np.std(dice_scores)
    mean_iou = np.mean(iou_scores)
    sd_iou = np.std(iou_scores)

    logging.info("Mean Dice Score: {:.4f} ± {:.4f}".format(mean_dice, sd_dice))
    logging.info("Mean IoU Score: {:.4f} ± {:.4f}".format(mean_iou, sd_iou))
