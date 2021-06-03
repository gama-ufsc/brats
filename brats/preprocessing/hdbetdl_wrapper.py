import os
import subprocess

from pathlib import Path

from hd_bet.model import predict
import nibabel as nib

import numpy as np

from skimage.transform import resize

def hd_bet_dl(in_file_fpath, out_prefix, in_file_flair_path, device=None, mode=None, tta=None, pp=None):
    in_file_fpath = Path(in_file_fpath)
    out_prefix_flair = Path(out_prefix + "_flair")
    out_prefix = Path(out_prefix)

    x = nib.load(in_file_fpath)
    
    affine = x.affine

    x = x.get_fdata()

    x = (x - x.mean())/(x.std() + 1e-6)

    mean = x.mean()
    std = x.std()

    x = resize(x, (240, 160, 160))

    brain, mask = predict(x)

    brain = brain*std + mean

    brain_nii = nib.Nifti1Image(brain, affine)
    mask_nii = nib.Nifti1Image(mask, affine)

    out_file_fpath = out_prefix.with_suffix('.nii.gz')
    mask_fpath = Path(str(out_prefix) + '_mask.nii.gz')

    nib.save(brain_nii, out_file_fpath)
    nib.save(mask_nii, mask_fpath)

    x = nib.load(in_file_flair_path)
    affine = x.affine
    x = x.get_fdata()

    flair_brain = x*mask
    brain_flair_nii = nib.Nifti1Image(flair_brain, affine)

    out_file_fpath_flair = out_prefix_flair.with_suffix('.nii.gz')

    nib.save(brain_flair_nii, out_file_fpath_flair)

    return out_file_fpath, out_file_fpath_flair, mask_fpath
