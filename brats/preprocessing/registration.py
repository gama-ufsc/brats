from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union

import nibabel as nib

from brats.preprocessing.base import Step


class RegistrationStep(Step,ABC):
    """Register the reference modality to another image using ANTs.

    Registers one of the modalities to the provided image (can be a template).
    They should be the same MRI modality unless you know what you are doing.
    Optionally, apply the generated transformation to the modalities.

    Attributes:
        reference_modality: modality to register to the target.
        target: modality of the target image to be registered to.
        apply: if True, applies the generated transformation to the modalities.
        num_threads: number of threads used by ANTs (see
        nipype.interfaces.ants.Registration).
    """
    def __init__(self, reference_modality: str,
        target: Union[str, nib.Nifti1Image], apply=False, tmpdir=None) -> None:
        super().__init__(tmpdir=tmpdir)

        self.reference_modality = reference_modality.lower()

        if isinstance(target, nib.Nifti1Image):
            self.target = target
        else:
            if target is not None:
                self.target = nib.load(target)

        self.apply = apply

    @abstractmethod
    def _register(self, modality: nib.Nifti1Image):
        """Wrapper that registers `modality` to target and returns the transform.

        Args:
            modality: loaded and stored (modality.get_filename() must work)
            modality that will be registered to target.

        Returns:
            transform_fpath: path to the transform that register modality to
            the target.
        """

    @abstractmethod
    def _apply(self, modality: nib.Nifti1Image, transform_fpath: str):
        """Wrapper that transforms `modality` to target space.

        Args:
            modality: loaded and stored (modality.get_filename() must work)
            modality that will be transformed.
            transform_fpath: path to the transform to be applied to `modality`.

        Returns:
            transf_image: path to the image after transformation.
        """

    def run(self, context: Dict) -> Dict:
        """Context must contain at lest the reference modality.
        """
        modalities = context['modalities'][-1]

        if self.apply:
            transf_modalities = dict()

        # generate transformation
        str_transform_fpath = self._register(
            modalities[self.reference_modality]
        )

        for mod, image in modalities.items():
            context['transforms'][mod].append(str_transform_fpath)
            if self.apply:
                # apply transformation to the modality
                transf_modalities[mod] = self._apply(image, str_transform_fpath)

        if self.apply:
            context['modalities'].append(transf_modalities)

        return context
