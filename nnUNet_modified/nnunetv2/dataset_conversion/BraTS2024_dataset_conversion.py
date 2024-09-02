import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img).astype(np.int32)  # Convert to integers

    uniques = np.unique(img_npy)
    print(f'Unique labels found: {uniques}')  # Fahim: added to check the unique labels
    for u in uniques:
        if u not in [0, 1, 2, 3, 4]:
            raise RuntimeError(f'unexpected label: {u}')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 1] = 1  # NETC
    seg_new[img_npy == 2] = 2  # SNFH
    seg_new[img_npy == 3] = 3  # ET
    seg_new[img_npy == 4] = 4  # RC

    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_labels_back_to_BraTS(seg: np.ndarray):
    seg = seg.astype(np.int32)  # Ensure labels are integers
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 1  # NETC
    new_seg[seg == 2] = 2  # SNFH
    new_seg[seg == 3] = 3  # ET
    new_seg[seg == 4] = 4  # RC
    return new_seg


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a).astype(np.int32)  # Convert to integers
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves them
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


if __name__ == '__main__':
    brats_data_dir = '/home/fahim/F2NetViT/Dataset/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2'

    task_id = 111
    task_name = "BraTS2024"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)
    for c in case_ids:
        shutil.copy(join(brats_data_dir, c, c + "-t1n.nii.gz"), join(imagestr, c.replace("-", "_") + '_0000.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t1c.nii.gz"), join(imagestr, c.replace("-", "_") + '_0001.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t2w.nii.gz"), join(imagestr, c.replace("-", "_") + '_0002.nii.gz'))
        shutil.copy(join(brats_data_dir, c, c + "-t2f.nii.gz"), join(imagestr, c.replace("-", "_") + '_0003.nii.gz'))

        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
            join(brats_data_dir, c, c + "-seg.nii.gz"),
            join(labelstr, c.replace("-", "_") + '.nii.gz')
        )

    generate_dataset_json(out_base,
                          channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'WT' : (1, 2, 3),
                              'TC': (1, 3,),
                              'resection cavity': (4,)

                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')
