import os
import subprocess

from pathlib import Path


def brainmage(in_file_fpath, mask_fpath, device):
    in_file_fpath = Path(in_file_fpath)
    mask_fpath = Path(mask_fpath)

    if device == 'gpu':
        device = 0

    cmd = f"brain_mage_single_run -i {in_file_fpath} -o {mask_fpath}"\
          f" -dev {device}"

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'  # to avoid mkl-service error

    _ = subprocess.run(cmd, shell=True, check=True)

    return mask_fpath
