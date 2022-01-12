import os
import subprocess

import nibabel as nib

from pathlib import Path
from typing import Dict, List

from dotenv import find_dotenv, load_dotenv

from .base import Step

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

try:
    _captk_cmd = str(Path(os.environ['CAPTK_PATH'])/'captk')
except KeyError:
    raise EnvironmentError(
        'Please be sure that `CAPTK_PATH` is set to the '
        'folder that contain the executable.'
    )


def captk_brats_pipeline(t1ce_fpath: str, t1_fpath: str, t2_fpath: str,
                         flair_fpath: str, tmpdir: str, subject_id: str = None,
                         skullstripping=False, brats=False):
    if subject_id is None:
        subject_id = Path(t1ce_fpath).name.replace('.nii', '').replace('.gz', '').replace('.dcm', '')

    out_dir = Path(tmpdir)/subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _captk_cmd + ' BraTSPipeline.cwl'

    cmd += f" -t1c {t1ce_fpath} -t1 {t1_fpath} -t2 {t2_fpath} -fl {flair_fpath}"
    cmd += f" -o {str(out_dir)}"
    cmd += f" -d 0 -s {int(skullstripping)} -b {int(brats)}"

    try:
        _ = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return False
    else:
        out = dict()
        images = dict()
        transfs = dict()
        # images['flair'] = out_dir/'FL_rai.nii.gz'
        # images['flair'] = out_dir/'FL_rai_n4.nii.gz'
        # images['flair'] = out_dir/'FL_raw.nii.gz'
        images['flair'] = out_dir/'FL_to_SRI.nii.gz'
        transfs['flair'] = out_dir/'FL_to_T1CE.mat'
        # images['t1ce'] = out_dir/'T1CE_rai.nii.gz'
        # images['t1ce'] = out_dir/'T1CE_rai_n4.nii.gz'
        # images['t1ce'] = out_dir/'T1CE_raw.nii.gz'
        images['t1ce'] = out_dir/'T1CE_to_SRI.nii.gz'
        transfs['t1ce'] = out_dir/'T1CE_to_SRI.mat'
        # images['t1'] = out_dir/'T1_rai.nii.gz'
        # images['t1'] = out_dir/'T1_rai_n4.nii.gz'
        # images['t1'] = out_dir/'T1_raw.nii.gz'
        images['t1'] = out_dir/'T1_to_SRI.nii.gz'
        transfs['t1'] = out_dir/'T1_to_T1CE.mat'
        # images['t2'] = out_dir/'T2_rai.nii.gz'
        # images['t2'] = out_dir/'T2_rai_n4.nii.gz'
        # images['t2'] = out_dir/'T2_raw.nii.gz'
        images['t2'] = out_dir/'T2_to_SRI.nii.gz'
        transfs['t2'] = out_dir/'T2_to_T1CE.mat'

        out['images'] = images
        out['transfs'] = transfs

        if skullstripping:
            # TODO: implement skullstripping output parsing
            raise NotImplementedError

        if brats:
            # TODO: implement tumor segmentation output parsing
            raise NotImplementedError

        return out

def captk_deepmedic(in_fpaths: List[str], out_dir: str,
                    deepmedic_model='skullStripping_modalityAgnostic'):
    model_fpath = Path(os.environ['DEEPMEDIC_PATH'])/'saved_models'/deepmedic_model
    cmd = _captk_cmd + ' DeepMedic'

    cmd += f" -i {','.join([str(f) for f in in_fpaths])}"
    cmd += f" -o {out_dir}"
    cmd += f" -md {model_fpath}"

    try:
        _ = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return False
    else:
        return Path(out_dir)/'predictions/testApiSession/predictions/Segm.nii.gz'

class BraTSRegistrationStep(Step):
    """Register images following the BraTS pipeline implemented in CaPTk.

    Input MUST contain t1, t2, t1ce and flair modalities.
    """
    def __init__(self, tmpdir=None) -> None:
        super().__init__(tmpdir=tmpdir)

    def run(self, context: Dict) -> Dict:
        modalities = context['modalities'][-1]

        out = captk_brats_pipeline(
            t1ce_fpath=modalities['t1ce'].get_filename(),
            t1_fpath=modalities['t1'].get_filename(),
            t2_fpath=modalities['t2'].get_filename(),
            flair_fpath=modalities['flair'].get_filename(),
            tmpdir=self.tmpdir,
            subject_id=None,
            skullstripping=False,
            brats=False,
        )

        context['modalities'].append({
            mod: nib.load(img_fpath) for mod, img_fpath in out['images'].items()
        })

        for mod in out['transfs'].keys():
            context['transforms'][mod].append(out['transfs']['t1ce'])
            if mod != 't1ce':  # t1ce is used a reference for this pipeline
                context['transforms'][mod].append(out['transfs'][mod])

        return context
