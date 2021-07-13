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
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    # seg_new[img_npy > 0] = 1
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

def make_nnunet_tcga_dataset(dataset_dir, target_images_dir, overwrite):
    target_images_dir = Path(target_images_dir)
    dataset_dir = Path(dataset_dir)

    patient_names = dataset_dir.glob('*.nii.gz')
    patient_names = {'_'.join(p.name.split('_')[:-1]) for p in patient_names}
    for patient_name in tqdm(patient_names):
        flair = Path(dataset_dir, patient_name + "_flair.nii.gz")
        t1 = Path(dataset_dir, patient_name + "_t1.nii.gz")
        t1c = Path(dataset_dir, patient_name + "_t1ce.nii.gz")
        t2 = Path(dataset_dir, patient_name + "_t2.nii.gz")

        assert all([
            flair.exists(),
            t1.exists(),
            # t1c.exists(),
            # t2.exists(),
        ]), "%s doesn't have all modalities" % patient_name

        new_flair = Path(target_images_dir, patient_name + "_0000.nii.gz")
        new_t1 = Path(target_images_dir, patient_name + "_0001.nii.gz")
        new_t1c = Path(target_images_dir, patient_name + "_0002.nii.gz")
        new_t2 = Path(target_images_dir, patient_name + "_0003.nii.gz")

        def _create_or_overwrite(symlink_path, dst_path, overwrite: bool):
            try:
                symlink_path.symlink_to(dst_path)
            except FileExistsError:
                if overwrite:
                    symlink_path.unlink()
                    symlink_path.symlink_to(dst_path)

        _create_or_overwrite(new_flair, flair, overwrite)
        _create_or_overwrite(new_t1, t1, overwrite)
        _create_or_overwrite(new_t1c, t1c, overwrite)
        _create_or_overwrite(new_t2, t2, overwrite)

    return patient_names

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
        flair = Path(patdir, patient_name + "_flair.nii.gz")
        t1 = Path(patdir, patient_name + "_t1.nii.gz")
        t1c = Path(patdir, patient_name + "_t1ce.nii.gz")
        t2 = Path(patdir, patient_name + "_t2.nii.gz")

        assert all([
            flair.exists(),
            t1.exists(),
            t1c.exists(),
            t2.exists(),
        ]), "%s doesn't have all modalities" % patient_name

        new_flair = Path(target_images_dir, patient_name + "_0000.nii.gz")
        new_t1 = Path(target_images_dir, patient_name + "_0001.nii.gz")
        new_t1c = Path(target_images_dir, patient_name + "_0002.nii.gz")
        new_t2 = Path(target_images_dir, patient_name + "_0003.nii.gz")

        def _create_or_overwrite(symlink_path, dst_path, overwrite: bool):
            try:
                symlink_path.symlink_to(dst_path)
            except FileExistsError:
                if overwrite:
                    symlink_path.unlink()
                    symlink_path.symlink_to(dst_path)

        _create_or_overwrite(new_flair, flair, True)
        _create_or_overwrite(new_t1, t1, True)
        _create_or_overwrite(new_t1c, t1c, True)
        _create_or_overwrite(new_t2, t2, True)

        if target_seg_dir is not None:
            seg = Path(patdir, patient_name + "_seg.nii.gz")
            assert seg.exists(), "%s has no segmentation" % patient_name
            copy_BraTS_segmentation_and_convert_labels(str(seg), str(Path(target_seg_dir, patient_name + ".nii.gz")))
    
    return patient_names
