from abc import ABC, abstractmethod
from typing import Dict

import nibabel as nib

from .base import Step


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

    @abstractmethod
    def _apply(self, modality: nib.Nifti1Image, brain_mask: nib.Nifti1Image
        ) -> nib.Nifti1Image:
        """Wrapper to apply `brain_mask` in `modality`.

        Returns:
            brain_modality: `modality` after masking operation, that is,
            (hopefully) without non-brain-tissues.    
        """

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
