from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union

import nibabel as nib
import numpy as np

from .base import Step


def apply_mask(image: Union[str, Path, nib.Nifti1Image],
               mask: Union[str, Path, nib.Nifti1Image],
               out_prefix: Union[str, Path]) -> nib.Nifti1Image:
    """Apply mask to image using matrix operations.

    Image and mask must contain same affine. Background is set to zero. Image
    is shifted to positive values.

    Args:
        image: image to apply the mask.
        mask: binary mask to be applied.
        out_prefix: prefix of the resulting image (including directory).
    """
    if image is not nib.Nifti1Image:
        image = nib.load(image)

    if mask is not nib.Nifti1Image:
        mask = nib.load(mask)
    
    out_prefix = Path(out_prefix)

    assert np.allclose(image.affine, mask.affine), (
        'image and mask have different affine matrices'
    )

    brain_data = image.get_fdata() * mask.get_fdata()
    brain_data -= np.min(brain_data)
    brain_data += 1
    brain_data[mask.get_fdata() == 0] = 0.0

    brain = nib.Nifti1Image(brain_data, image.affine, image.header)

    if out_prefix.is_dir():
        brain_fpath = out_prefix / Path(image.get_filename()).name
    else:
        brain_fpath = Path(str(out_prefix) + Path(image.get_filename()).name)
    nib.save(brain, brain_fpath)

    return brain

class BrainExtraction(Step,ABC):
    """Perform brain extraction on all modalities.

    Generates brain mask from `bet_modality` image and (optionally) applies the
    mask in the other modalities.

    Attributes:
        bet_modality: modality upon which to generate the brain mask.
        apply: wheter to apply the mask to the image (default) or not.
    """
    def __init__(self, bet_modality: str, apply=True, tmpdir=None) -> None:
        super().__init__(tmpdir=tmpdir)

        self.bet_modality = bet_modality.lower()

        self.apply = apply

    @abstractmethod
    def _bet(self, modality: nib.Nifti1Image) -> nib.Nifti1Image:
        """Wrapper to the brain mask generation.

        Args:
            modality: image upon which to generate the brain mask.
        
        Returns:
            brain_mask: mask of the brain in the space of `modality`.
        """

    def _apply(self, modality: nib.Nifti1Image, brain_mask: nib.Nifti1Image
        ) -> nib.Nifti1Image:
        """Wrapper to apply `brain_mask` in `modality`.

        See .apply_mask

        Returns:
            brain_modality: `modality` after masking operation, that is,
            (hopefully) without non-brain-tissues.    
        """
        return apply_mask(modality, brain_mask, self.tmpdir/'brain_')

    def run(self, context: Dict) -> Dict:
        modalities = context['modalities'][-1]

        brain_mask = self._bet(modalities[self.bet_modality])

        context['brain_mask'] = brain_mask

        if self.apply:
            brain_modalities = dict()
            for mod, image in modalities.items():
                brain_modalities[mod] = self._apply(image, brain_mask)

            context['modalities'].append(brain_modalities)

        return context
