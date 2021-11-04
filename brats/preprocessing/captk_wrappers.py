import os
import subprocess

from pathlib import Path
from typing import List

from dotenv import find_dotenv, load_dotenv

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

_captk_cmd = str(Path(os.environ['CAPTK_PATH'])/'captk')
_greedy_cmd = str(Path(os.environ['GREEDY_PATH'])/'greedy')


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

def greedy_apply_transforms(moving_fpath: str, output_fpath: str, transforms_fpaths: List[str], interpolation='LINEAR'):
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
