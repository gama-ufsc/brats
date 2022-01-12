import os
import subprocess

from pathlib import Path

import nibabel as nib

from .brainx import BrainExtraction


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

class HDBET(BrainExtraction):
    """Apply pre-trained HD-BET.

    Attributes:
        device, mode, tta, pp: see hd-bet executable.
        apply: wheter to apply the mask to the image (default) or not.
    """
    def __init__(self, bet_modality: str, device=None, mode=None, tta=None,
        pp=None, apply=True, tmpdir=None) -> None:
        super().__init__(bet_modality, apply=apply, tmpdir=tmpdir)

        self.hd_bet_device = device
        self.hd_bet_mode = mode
        self.hd_bet_tta = tta
        self.hd_bet_pp = pp

    def _bet(self, modality: nib.Nifti1Image) -> nib.Nifti1Image:
        mod_name = Path(modality.get_filename()).name.replace('.nii', '') \
                                                     .replace('.gz', '')
        _, brain_mask_fpath = hd_bet(
            modality.get_filename(),
            self.tmpdir/f"{mod_name}_mask",
            device=self.hd_bet_device,
            mode=self.hd_bet_mode,
            tta=self.hd_bet_tta,
            pp=self.hd_bet_pp,
        )

        return nib.load(brain_mask_fpath)
