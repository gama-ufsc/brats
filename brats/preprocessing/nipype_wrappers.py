import os

from nipype.interfaces import ants, fsl
from nipype.interfaces.freesurfer import WatershedSkullStrip, Normalize


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


def freesurfer_normalize(in_file_fpath, out_prefix: str) -> str:
    """Run FreeSurfer's normalization routine.
    """
    norm = Normalize()

    norm.inputs.in_file = str(in_file_fpath)
    norm.inputs.out_file = out_prefix + '_interim_' + os.path.basename(in_file_fpath)

    norm_res = norm.run()

    return norm_res.outputs.out_file


def freesurfer_bet(in_file_fpath, out_prefix: str, brain_atlas: bool = False,
                   normalize: bool = False) -> str:
    """Run FreeSurfer's watershed algorithm for skull stripping.
    """
    if normalize:
        _in_file = freesurfer_normalize(in_file_fpath, out_prefix + '_norm_')
    else:
        _in_file = str(in_file_fpath)

    sks = WatershedSkullStrip()

    sks.inputs.in_file = _in_file
    sks.inputs.out_file = out_prefix + os.path.basename(in_file_fpath)

    if brain_atlas:
        sks.inputs.args = '-atlas'

    try:
        res = sks.run()
    except RuntimeError:
        sks.inputs.in_file = freesurfer_normalize(in_file_fpath,
                                                  out_prefix + '_norm_')

        res = sks.run()

    return res.outputs.out_file
