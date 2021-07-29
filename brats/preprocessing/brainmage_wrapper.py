import os
import subprocess

from pathlib import Path


def brainmage_multi4(t1_fpath, t2_fpath, t1ce_fpath, flair_fpath, mask_fpath, device):
    mask_fpath = Path(mask_fpath)

    in_file_fpaths = list()
    for fpath in [t1_fpath, t2_fpath, t1ce_fpath, flair_fpath]:
        in_file_fpaths.append(str(fpath))
    in_file_fpaths = ','.join(in_file_fpaths)

    if device == 'gpu':
        device = 0
    elif type(device) != int and device.lower() != 'cpu':
        raise ValueError("`device` must be 'gpu', 'cpu', or the gpu id (int)")

    cmd = f"brain_mage_single_run_multi_4 -i {in_file_fpaths} -o {mask_fpath}"\
          f" -dev {device}"

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'  # to avoid mkl-service error

    print(cmd)
    _ = subprocess.run(cmd, shell=True, check=True)

    return mask_fpath

def brainmage(in_file_fpath, mask_fpath, device):
    in_file_fpath = Path(in_file_fpath)
    mask_fpath = Path(mask_fpath)

    if device == 'gpu':
        device = 0
    elif type(device) != int and device.lower() != 'cpu':
        raise ValueError("`device` must be 'gpu', 'cpu', or the gpu id (int)")

    cmd = f"brain_mage_single_run -i {in_file_fpath} -o {mask_fpath}"\
          f" -dev {device}"

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'  # to avoid mkl-service error

    print(cmd)
    _ = subprocess.run(cmd, shell=True, check=True)

    return mask_fpath
