import os
import subprocess

from pathlib import Path
from typing import Dict, List

import nibabel as nib

from dotenv import find_dotenv, load_dotenv

from .registration import RegistrationStep

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

try:
    _greedy_cmd = str(Path(os.environ['GREEDY_PATH'])/'greedy')
except KeyError:
    raise EnvironmentError(
        'Please be sure that `GREEDY_PATH` is set to the '
        'folder that contain the executable.'
    )


def greedy_apply_transforms(moving_fpath: str, output_fpath: str,
    transforms_fpaths: List[str], interpolation='LINEAR'):
    atlas_fpath = Path(os.environ['GREEDY_PATH'])/'../data/sri24/atlastImage.nii.gz'

    transforms = [str(f) for f in transforms_fpaths]

    cmd = _greedy_cmd
    cmd += f" -d 3 -rf {str(atlas_fpath)} -ri LINEAR"
    cmd += f" -ri {interpolation} -rm {moving_fpath} {output_fpath}"
    cmd += f" -r {' '.join(transforms)}"

    try:
        _ = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return False
    else:
        return output_fpath

def greedy_registration(fixed_image_fpath: str, moving_image_fpath: str,
                        out_mat_fpath: str):
    cmd = _greedy_cmd
    cmd += f" -d 3 -a -m NMI"
    cmd += f" -i {fixed_image_fpath} {moving_image_fpath}"
    cmd += f" -o {out_mat_fpath}"
    cmd += f" -ia-image-centers -n 100x50x100 -dof 6"

    try:
        _ = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return False
    else:
        return out_mat_fpath

class GreedyRegistrationStep(RegistrationStep):
    """Register the reference modality to another image using greedy.

    Registers one of the modalities to the provided image (can be a template).
    They should be the same MRI modality unless you know what you are doing.
    Optionally, apply the generated transformation to the modalities.

    Attributes:
        reference_modality: modality to register to the target.
        target: modality of the target image to be registered to.
        apply: if True, applies the generated transformation to the modalities.
    """
    def _register(self, modality: nib.Nifti1Image):
        return greedy_registration(
            self.target.get_filename(),
            modality.get_filename(),
            str(self.tmpdir/f"str_transform.mat"),
        )

    def _apply(self, modality: nib.Nifti1Image, transform_fpath: str):
        target_name = Path(self.target.get_filename()).name.split('.')[0]
        mod_name = Path(modality.get_filename()).name.split('.')[0]

        transf_image_fpath = greedy_apply_transforms(
            modality.get_filename(),
            str(self.tmpdir/f"{mod_name}_at_{target_name}.nii.gz"),
            [transform_fpath],
        )

        return nib.load(transf_image_fpath)
