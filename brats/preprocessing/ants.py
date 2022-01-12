import os

import nibabel as nib

from pathlib import Path
from typing import Dict, Union

from nipype.interfaces import ants

from .registration import Registration


def ants_n4bfc(input_fpath, output_fpath, num_threads: int = -1):
    n4 = ants.N4BiasFieldCorrection()

    n4.inputs.input_image = str(input_fpath)
    n4.inputs.copy_header = True
    n4.inputs.save_bias = False
    n4.inputs.num_threads = num_threads
    n4.inputs.output_image = str(output_fpath)

    res = n4.run()

    return res.outputs.output_image

def ants_registration(fixed_image_fpath, moving_image_fpath,
                      transformation_prefix: str, num_threads=-1) -> tuple:
    """Run ANTS registration to generate the alignment transformation.
    """
    reg = ants.Registration()

    reg.inputs.fixed_image = fixed_image_fpath
    reg.inputs.moving_image = moving_image_fpath
    reg.inputs.num_threads = num_threads

    reg.inputs.output_transform_prefix = transformation_prefix

    # registration parameters
    reg.inputs.collapse_output_transforms = True
    reg.inputs.float = True
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.initial_moving_transform_com = 0
    reg.inputs.interpolation = 'BSpline'
    reg.inputs.interpolation_parameters = (3,)
    reg.inputs.metric = ['Mattes']
    reg.inputs.metric_weight = [1]
    reg.inputs.number_of_iterations = [[1000, 500, 250]]
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.sampling_strategy = ['Regular']
    reg.inputs.sampling_percentage = [0.5]
    reg.inputs.shrink_factors = [[8, 4, 2]]
    reg.inputs.sigma_units = ['vox']
    reg.inputs.smoothing_sigmas = [[3, 2, 1]]
    reg.inputs.transforms = ['Rigid']
    reg.inputs.transform_parameters = [(0.1,)]
    reg.inputs.use_estimate_learning_rate_once = [True]

    res = reg.run()

    return res.outputs.forward_transforms[0], res.outputs.reverse_transforms[0]

def ants_transformation(input_image_fpath, ref_image_fpath, transforms_fpaths,
                        output_prefix: str, num_threads=-1,
                        interpolation='Linear', reverse=False) -> str:
    """Run ANTS transformation to apply `transforms` to `input_image`.
    """
    apply = ants.ApplyTransforms()

    apply.inputs.input_image = input_image_fpath
    apply.inputs.reference_image = ref_image_fpath
    apply.inputs.transforms = transforms_fpaths
    apply.inputs.num_threads = num_threads

    output_image_fpath = output_prefix + os.path.basename(input_image_fpath)
    apply.inputs.output_image = output_image_fpath

    apply.inputs.dimension = 3
    apply.inputs.float = True

    apply.inputs.interpolation = interpolation

    apply.inputs.invert_transform_flags = [reverse for _ in transforms_fpaths]

    res = apply.run()

    return res.outputs.output_image

class ANTsRegistration(Registration):
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
        target: Union[str, nib.Nifti1Image], apply=False, num_threads=-1,
        tmpdir=None) -> None:
        super().__init__(reference_modality, target, apply=apply, tmpdir=tmpdir)

        self.num_threads = num_threads

    def _register(self, modality: nib.Nifti1Image):
        transform_fpath, _ = ants_registration(
            self.target.get_filename(),
            modality.get_filename(),
            str(self.tmpdir/f"str_transform_"),
            self.num_threads,
        )

        return transform_fpath

    def _apply(self, modality: nib.Nifti1Image, transform_fpath: str):
        target_name = Path(self.target.get_filename()).name.split('.')[0]

        transf_image_fpath = ants_transformation(
            modality.get_filename(),
            self.target.get_filename(),
            [transform_fpath],
            str(self.tmpdir/f"at_{target_name}_"),
            self.num_threads,
        )

        return nib.load(transf_image_fpath)

class ANTsWithinSubjectRegistration(ANTsRegistration):
    """Register multiple modalities into the space of one of them using ANTs.

    Effectively, the resulting modalities are all aligned to the modality taken
    as reference.

    Attributes:
        reference_modality: modality to register the others to.
        apply: if True, applies the generated transformation to the modalities.
        num_threads: number of threads used by ANTs (see
        nipype.interfaces.ants.Registration).
    """
    def __init__(self, reference_modality: str, apply=False, num_threads=-1,
        tmpdir=None) -> None:
        super().__init__(reference_modality, None, apply=apply,
                         num_threads=num_threads, tmpdir=tmpdir)

    def run(self, context: Dict) -> Dict:
        """Context must contain at lest the reference modality and one other.
        """
        modalities = context['modalities'][-1]

        self.target = modalities[self.reference_modality]

        if self.apply:
            transf_modalities = dict()

        for mod, image in modalities.items():
            if mod != self.reference_modality:
                # generate transformation using ANTs registration
                wsr_transform_fpath = self._register(image)
                context['transforms'][mod].append(wsr_transform_fpath)

                if self.apply:
                    # apply transformation to the modality
                    transf_modalities[mod] = self._apply(image,
                                                         wsr_transform_fpath)
            else:
                if self.apply:
                    # as the reference modality is already in the target space,
                    # no transformation is required
                    transf_modalities[mod] = image

        if self.apply:
            context['modalities'].append(transf_modalities)

        return context
