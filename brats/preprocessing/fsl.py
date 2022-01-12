import os

from pathlib import Path
from typing import Dict

import nibabel as nib

from nipype.interfaces.fsl.utils import Reorient2Std
from nipype.interfaces import fsl

from brats.preprocessing.brainx import BrainExtraction

from .base import Step


def fsl_bet(in_file_fpath, out_prefix: str, fast=False) -> str:
    """Run BET algorithm from FSL to remove non-brain content from image.

    if `fast` is false (default), runs reduced bias BET variant (much slower).
    """
    bet = fsl.BET()
    bet.inputs.in_file = str(in_file_fpath)
    bet.inputs.mask = False

    bet.inputs.frac = 0.50

    bet.inputs.mask = True

    if not fast:
        bet.inputs.reduce_bias = True

    bet.inputs.no_output = False

    # handle .nii x .nii.gz output file creation
    out_file = out_prefix + os.path.basename(in_file_fpath)
    if out_file[-3:] == 'nii':
        out_file += '.gz'
    bet.inputs.out_file = out_file

    # print(bet.cmdline)
    # # bypass mutually exclusive restriction on bet variations from nipype's
    # # implementation, following the parameters used by Isensee et al., at
    # # HD-BET paper
    # cmdline = bet.cmdline
    # cmdline += ' -R -S -B'
    # subprocess.run(cmdline.split(' '))

    res = bet.run()

    return res.outputs.out_file, res.outputs.mask_file


def fsl_applymask(in_file_fpath, mask_file_fpath, out_prefix):
    apply = fsl.ApplyMask()
    apply.inputs.in_file = str(in_file_fpath)
    apply.inputs.mask_file = str(mask_file_fpath)

    if str(in_file_fpath).endswith('.nii'):
        apply.inputs.output_type = 'NIFTI'
    elif str(in_file_fpath).endswith('.nii.gz'):
        apply.inputs.output_type = 'NIFTI_GZ'

    apply.inputs.out_file = str(out_prefix) + os.path.basename(in_file_fpath)

    res = apply.run()

    return res.outputs.out_file

class StdReorientation(Step):
    """Reorient all modalities to the standard orientation.

    See nipype.interfaces.fsl.utils.Reorient2Std.
    """
    def run(self, context: Dict) -> Dict:
        modalities = context['modalities'][-1]

        out_modalities = dict()
        for mod, image in modalities.items():
            mod_name = Path(image.get_filename()).name

            reor = Reorient2Std()
            reor.inputs.in_file = image.get_filename()
            reor.inputs.out_file = self.tmpdir/('reoriented_'+mod_name)

            res = reor.run()

            out_modalities[mod] = nib.load(res.outputs.out_file)
        
        context['modalities'].append(out_modalities)

        return context

class BET(BrainExtraction):
    """Apply FSL's BET on all modalities.
    """
    def __init__(self, bet_modality: str, fast=True, apply=True, tmpdir=None
        ) -> None:
        super().__init__(bet_modality, apply=apply, tmpdir=tmpdir)

        self.fast = fast

    def _bet(self, modality: nib.Nifti1Image) -> nib.Nifti1Image:
        """Extract brain mask from `modality` using FSL's BET.
        """
        _, brain_mask_fpath = fsl_bet(
            modality.get_filename(),
            str(self.tmpdir/f"brain_"),
            fast=self.fast
        )

        return nib.load(brain_mask_fpath)

    def _apply(self, modality: nib.Nifti1Image, brain_mask: nib.Nifti1Image
        ) -> nib.Nifti1Image:
        brain_modality_fpath = fsl_applymask(
            modality.get_filename(),
            brain_mask.get_filename(),
            str(self.tmpdir/'brain_')
        )

        return nib.load(brain_modality_fpath)
