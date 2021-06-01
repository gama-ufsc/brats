# -*- coding: utf-8 -*-
from batchgenerators.utilities.file_and_folder_operations import *
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from tqdm import tqdm


# FROM nnunet.dataset_conversion !!!
def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    # seg_new[img_npy == 4] = 3
    # seg_new[img_npy == 2] = 1
    # seg_new[img_npy == 1] = 2
    seg_new[img_npy > 0] = 1
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

def make_nnunet_lemon_dataset(lemon_dir, target_images_dir):
    target_images_dir = Path(target_images_dir)
    lemon_dir = Path(lemon_dir)

    patient_names = lemon_dir.glob('*.nii.gz')
    patient_names = [p.name.split('_')[0] for p in patient_names]
    patient_names = list(set(patient_names))
    for patient_name in tqdm(patient_names):
        t1 = Path(lemon_dir, patient_name + "_T1w.nii.gz")
        flair = Path(lemon_dir, patient_name + "_FLAIR.nii.gz")

        assert all([
            t1.exists(),
            flair.exists(),
        ]), "%s has no t1+flair" % patient_name

        new_flair = Path(target_images_dir, patient_name + "_0000.nii.gz")
        new_flair.symlink_to(flair)
        new_t1 = Path(target_images_dir, patient_name + "_0001.nii.gz")
        new_t1.symlink_to(t1)

    return patient_names

def make_nnunet_dataset(patient_dirs, target_images_dir, target_seg_dir=None):
    target_images_dir = Path(target_images_dir)

    if target_seg_dir is not None:
        target_seg_dir = Path(target_seg_dir)

    patient_names = list()
    for patdir in tqdm(patient_dirs):
        if patdir.is_file():
            continue

        patient_name = patdir.name
        patient_names.append(patient_name)
        t1 = Path(patdir, patient_name + "_t1.nii.gz")
        flair = Path(patdir, patient_name + "_flair.nii.gz")

        assert all([
            t1.exists(),
            flair.exists(),
        ]), "%s has no t1+flair" % patient_name

        new_flair = Path(target_images_dir, patient_name + "_0000.nii.gz")
        new_flair.symlink_to(flair)
        new_t1 = Path(target_images_dir, patient_name + "_0001.nii.gz")
        new_t1.symlink_to(t1)

        if target_seg_dir is not None:
            seg = Path(patdir, patient_name + "_seg.nii.gz")
            assert seg.exists(), "%s has no segmentation" % patient_name
            copy_BraTS_segmentation_and_convert_labels(str(seg), str(Path(target_seg_dir, patient_name + ".nii.gz")))
    
    return patient_names
