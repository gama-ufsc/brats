import os
import subprocess

from pathlib import Path


def hd_bet(in_file_fpath, out_prefix, device=None, mode=None, tta=None, pp=None):
    in_file_fpath = Path(in_file_fpath)
    out_prefix = Path(out_prefix)

    cmd = f"hd-bet -i {in_file_fpath} -o {out_prefix}"

    if mode is not None:
        cmd += f" -mode {mode}"

    if device is not None:
        if device == 'gpu':
            device = 0
        cmd += f" -device {device}"

    if tta is not None:
        cmd += f" -tta {tta}"

    if pp is not None:
        cmd += f" -pp {pp}"

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'  # to avoid mkl-service error

    _ = subprocess.run(cmd, shell=True, check=True)

    out_file_fpath = Path(str(out_prefix) + '.nii.gz')
    mask_fpath = Path(str(out_prefix) + '_mask.nii.gz')

    return out_file_fpath, mask_fpath
