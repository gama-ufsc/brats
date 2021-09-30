from abc import ABC, abstractmethod
from logging import warn
from time import time
import os

from typing import List, Tuple, Dict, Union
from pathlib import Path

import nibabel as nib
import numpy as np

from scipy.ndimage import binary_fill_holes as fill_holes

from .bet import bet as our_bet
from .hdbet_wrapper import hd_bet
from .brainmage_wrapper import brainmage, brainmage_multi4
from .nipype_wrappers import ants_registration, ants_transformation, fsl_bet, fsl_applymask, freesurfer_bet


def apply_mask_match_brats(image_fpath: Union[str, Path],
                           mask_fpath: Union[str, Path], out_prefix: str) -> Path:
    image_fpath = Path(image_fpath)
    mask_fpath = Path(mask_fpath)
    out_prefix = Path(out_prefix)

    image = nib.load(image_fpath)
    mask = nib.load(mask_fpath)

    brain_data = image.get_fdata() * mask.get_fdata()
    brain_data -= np.min(brain_data)
    brain_data += 1
    brain_data[mask.get_fdata() == 0] = 0.0

    brain = nib.Nifti1Image(brain_data, image.affine, image.header)

    if out_prefix.is_dir():
        brain_fpath = out_prefix / image_fpath.name
    else:
        brain_fpath = Path(str(out_prefix) + image_fpath.name)
    nib.save(brain, brain_fpath)

    return Path(brain.get_filename())


class Preprocessor(ABC):
    """Preprocessing module of BraTS pipeline.

    Aligns the FLAIR and T1 modalities to a T1 template using ANTs functions.
    Also performs brain extraction (BET defined by child).
    """
    def __init__(self, template_fpath: str, tmpdir: str, bet_modality='T1',
                 bet_first=False, num_threads=-1, device='gpu'):
        assert os.path.exists(template_fpath), (
            '`template` must be a valid path to the template image'
        )
        self.template_fpath = template_fpath

        self.num_threads = num_threads

        self.bet_modality = bet_modality.lower()

        self.bet_first = bet_first

        assert device.lower() in ['cpu', 'gpu'], 'Device not recognized'
        self.device = device.lower()

        # tracks how long the bets took
        self.bet_cost_hist = list()

        # create directory for temporary files
        os.makedirs(tmpdir, exist_ok=True)
        self.tmpdir = tmpdir

    @staticmethod
    def registration(modalities: Dict[str,str], template_fpath: str,
                     tmpdir: str, num_threads=-1) -> Tuple[Dict[str,str],
                                                           Dict[str,List[str]]]:
        """Register modalities to template using T1 as reference.

        Args:
            modalities: dict containing the fpath of each modality to be
            registered.

        Returns:
            modalities_at_template: similar to `modalities`, but with them at
            the template's space.
            transforms: dict containing the transforms of each modality from
            their original space to the template space.
        """
        modalities_at_template = dict()

        transforms = dict()
        for mod in modalities:
            transforms[mod] = list()

        # t1 (subject) to template registration
        str_transform_fpath, _ = ants_registration(
            template_fpath,
            modalities['t1'],
            os.path.join(tmpdir, 'str_transform_'),
            num_threads,
        )

        # within-subject registration
        for mod, mod_fpath in modalities.items():
            transforms[mod].append(str_transform_fpath)

            if mod != 't1':
                wsr_transform_fpath, _ = ants_registration(
                    modalities['t1'],
                    mod_fpath,
                    os.path.join(tmpdir, 'wsr_transform_'),
                    num_threads,
                )
                transforms[mod].append(wsr_transform_fpath)

            # apply transformations
            modalities_at_template[mod] = ants_transformation(
                mod_fpath,
                template_fpath,
                transforms[mod],
                os.path.join(tmpdir, mod+'_template_'),
                num_threads,
            )

        return modalities_at_template, transforms

    @abstractmethod
    def _bet(self, modality_fpath) -> str:
        """ BETs `modality_fpath`, returning brain image and mask.

        UPDATE `self.bet_cost_hist`!
        """

    def bet_transform_apply(self, src_fpath, dst_fpath, transforms):
        """ BETs `src_fpath`, transforms the masks and applies in `dst_fpath`.
        """
        raw_brain_mask_fpath = self._bet(src_fpath)

        # transform mask to template space
        brain_mask_fpath = ants_transformation(
            raw_brain_mask_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 'mask_template_'),
            self.num_threads,
        )

        return brain_mask_fpath

    def bet(self, modalities, modalities_at_template, transforms
           ) -> Dict[str,str]:
        """Perform brain extraction given object parameters.

        Args:
            modalities: dict of paths to the raw modalities (see `self.run`).
            modalities_at_tempalte: dict of paths to the modalities at template
            space (see `self.registration`).
            transforms: dict of the transforms for each modality from their
            original space to the template space (see `self.registration`).
        """
        modalities_brain = dict()
        if self.bet_modality in modalities_at_template:
            if self.bet_first:
                brain_mask_fpath = self.bet_transform_apply(
                    modalities[self.bet_modality],
                    modalities_at_template[self.bet_modality],
                    transforms[self.bet_modality]
                )
            else:
                brain_mask_fpath = self._bet(
                    modalities_at_template[self.bet_modality]
                )

            for mod in modalities_at_template:
                modalities_brain[mod] = apply_mask_match_brats(
                    modalities_at_template[mod],
                    brain_mask_fpath,
                    os.path.join(self.tmpdir, 'brain_')
                )
        elif self.bet_modality == 'all':
            for mod in modalities_at_template:
                if self.bet_first:
                    brain_mask_fpath = self.bet_transform_apply(
                        modalities[mod],
                        modalities_at_template[mod],
                        transforms[mod]
                    )
                else:
                    brain_mask_fpath = self._bet(
                        modalities_at_template[mod]
                    )
                modalities_brain[mod] = apply_mask_match_brats(
                    modalities_at_template[mod],
                    brain_mask_fpath,
                    os.path.join(self.tmpdir, 'brain_')
                )
        else:
            raise AttributeError(f'`{self.bet_modality}` is not a valid'
                                  ' reference for betting.')

        return modalities_brain, brain_mask_fpath

    def run(self, flair_fpath: str = None, t1_fpath: str = None,
            t1ce_fpath: str = None, t2_fpath: str = None
            ) -> Tuple[Dict[str,nib.Nifti1Image], Dict[str,List[str]]]:
        """Run preprocessing pipeline.

        Args:
            *_fpath: path to nifti image containing the * modality of the
            subject. Main object of the preprocessing operations.

        Returns:
            images: dict containing preprocessed and loaded modalities.
            reverse_transform: Path to matrices that reverse the raw to
            preprocessed transformation.
        """
        modalities = dict()

        for mod in ['t1', 'flair', 't2', 't1ce']:
            mod_fpath = eval(mod+'_fpath')
            if mod_fpath is not None:
                modalities[mod] = mod_fpath
            else:
                if self.bet_modality == mod:
                    raise ValueError(
                        'At least the reference modality for'
                        ' betting must be provided'
                    )

        modalities_at_template, transforms = self.registration(
            modalities,
            self.template_fpath,
            self.tmpdir,
            self.num_threads,
        )

        modalities_brain, _ = self.bet(modalities, modalities_at_template,
                                       transforms)

        # load images
        modalities_brain = {mod: nib.load(mod_fpath)
                            for mod, mod_fpath in modalities_brain.items()}

        return modalities_brain, transforms

    def transform_prediction(self, pred: nib.Nifti1Image, transforms: list,
                             original_image: str) -> nib.Nifti1Image:
        """Transform prediction to original image's space.
        """
        # save pred file in case it is not yet
        if pred.get_filename() is None:
            nib.save(pred, os.path.join(self.tmpdir, 'pred.nii.gz'))

        # make sure the file exists
        assert pred.get_filename() is not None, (
            'saving of temporary prediction file failed, check whether the'
            ' temporary directory ({}) is available'.format(self.tmpdir)
        )

        # transforms using ants operation
        transformed_pred_image = ants_transformation(
            pred.get_filename(),
            original_image,
            transforms,
            os.path.join(self.tmpdir, 'transf_pred_'),
            self.num_threads,
            interpolation='NearestNeighbor',
            reverse=True,
        )

        return nib.load(transformed_pred_image)


class PreprocessorFreeSurfer(Preprocessor):
    """Preprocessing module of BraTS pipeline using FreeSurfer's skullstripper.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.
    """
    def __init__(self, template_fpath: str, tmpdir: str, fast_bet=True,
                 bet_modality='T1', bet_first=False, num_threads=-1,
                 device='cpu'):
        assert device.lower() == 'cpu', 'only cpu supported by FreeSurfer'

        super().__init__(template_fpath, tmpdir, bet_modality=bet_modality,
                         bet_first=bet_first, num_threads=num_threads,
                         device=device)

        self.fast_bet = fast_bet

    def _bet(self, modality_fpath):
        s_time = time()
        # extract brain
        brain_fpath = freesurfer_bet(
            modality_fpath,
            os.path.join(self.tmpdir, 'brain_'),
            brain_atlas=not self.fast_bet,
        )
        f_time = time()

        self.bet_cost_hist.append(f_time - s_time)

        # mask generation
        brain = nib.load(brain_fpath)

        mask_data = fill_holes(brain.get_fdata() != 0)

        mask = nib.Nifti1Image(mask_data, brain.affine, brain.header)

        brain_mask_fpath = os.path.join(self.tmpdir,
                                        'mask_'+os.path.basename(brain_fpath))
        nib.save(mask, brain_mask_fpath)

        return brain_mask_fpath


class PreprocessorFSL(Preprocessor):
    """Preprocessing module of BraTS pipeline using FSL's BET.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.
    """
    def __init__(self, template_fpath: str, tmpdir: str, fast_bet=True,
                 bet_modality='T1', bet_first=False, num_threads=-1,
                 device='cpu'):
        assert device.lower() == 'cpu', 'only cpu supported by FSL BET'

        super().__init__(template_fpath, tmpdir, bet_modality=bet_modality,
                         bet_first=bet_first, num_threads=num_threads,
                         device=device)

        self.fast_bet = fast_bet

    def _bet(self, modality_fpath):
        s_time = time()
        # extract brain
        _, brain_mask_fpath = fsl_bet(
            modality_fpath,
            os.path.join(self.tmpdir, 'brain_'),
            fast=self.fast_bet
        )
        f_time = time()

        self.bet_cost_hist.append(f_time - s_time)

        return brain_mask_fpath


class PreprocessorHDBET(Preprocessor):
    """Preprocessing module of BraTS pipeline using HD-BET.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.
    """
    def __init__(self, template_fpath: str, tmpdir: str, bet_modality='FLAIR',
                 bet_first=False, num_threads=-1, device='gpu', **hdbet_kwargs):
        super().__init__(template_fpath, tmpdir, bet_modality=bet_modality,
                         bet_first=bet_first, num_threads=num_threads,
                         device=device)

        self.hdbet_kwargs = hdbet_kwargs

    def _bet(self, modality_fpath):
        s_time = time()
        _, brain_mask_fpath = hd_bet(
            modality_fpath,
            os.path.join(
                self.tmpdir,
                os.path.basename(modality_fpath).replace('.nii', '') \
                                                .replace('.gz', '')) \
                                                + '_mask',
            device=self.device,
            **self.hdbet_kwargs
        )
        f_time = time()

        self.bet_cost_hist.append(f_time - s_time)

        return brain_mask_fpath


class PreprocessorBrainMaGe(Preprocessor):
    """Preprocessing module of BraTS pipeline using BrainMaGe.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.
    """
    def __init__(self, template_fpath: str, tmpdir: str, bet_modality='T1',
                 mode='ma', bet_first=False, num_threads=-1, device='gpu'):

        super().__init__(template_fpath, tmpdir, bet_modality=bet_modality,
                         bet_first=bet_first, num_threads=num_threads,
                         device=device)

        assert mode.lower() in ['ma', 'multi4'], f'{mode} is not supported (only `ma` or `multi4`)'
        self.mode = mode.lower()

    def bet(self, modalities, modalities_at_template, transforms
           ) -> Dict[str,str]:
        """Perform brain extraction given object parameters.
        
        Args:
            modalities: dict of paths to the raw modalities (see `self.run`).
            modalities_at_tempalte: dict of paths to the modalities at template
            space (see `self.registration`).
            transforms: dict of the transforms for each modality from their
            original space to the template space (see `self.registration`).
        """
        if self.mode != 'multi4':
            return super().bet(modalities, modalities_at_template, transforms)
        else:
            if self.bet_first:
                _modalities = {m+'_fpath':fp for m, fp in modalities.items()}
            else:
                _modalities = {m+'_fpath':fp
                               for m, fp in modalities_at_template.items()}

            brain_mask_fpath = self._bet_multi_4(**_modalities)

            modalities_brain = dict()
            ms = set(modalities.keys()).union(set(modalities_at_template.keys()))
            for m in ms:
                mod_fpath = _modalities[m+'_fpath']
                modalities_brain[m] = apply_mask_match_brats(
                    mod_fpath,
                    brain_mask_fpath,
                    os.path.join(self.tmpdir, 'brain_'),
                )

            return modalities_brain, brain_mask_fpath

    def _bet_multi_4(self, t1_fpath, t2_fpath, t1ce_fpath, flair_fpath):
        brain_mask_fpath = os.path.join(
            self.tmpdir,
            'mask_' + os.path.basename(t1_fpath)
        )
        s_time = time()
        brain_mask_fpath = brainmage_multi4(
            t1_fpath,
            t2_fpath,
            t1ce_fpath,
            flair_fpath,
            brain_mask_fpath,
            device=self.device
        )
        f_time = time()

        self.bet_cost_hist.append(f_time - s_time)

        return brain_mask_fpath

    def _bet(self, modality_fpath):
        brain_mask_fpath = os.path.join(
            self.tmpdir,
            'mask_' + os.path.basename(modality_fpath)
        )
        s_time = time()
        brain_mask_fpath = brainmage(modality_fpath, brain_mask_fpath,
                                     device=self.device)
        f_time = time()

        self.bet_cost_hist.append(f_time - s_time)

        return brain_mask_fpath


class PreprocessorOurBET(Preprocessor):
    """Preprocessing module of BraTS pipeline using BrainMaGe.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.
    """
    def __init__(self, template_fpath: str, weights_fpath: str, tmpdir: str,
                 bet_modality='FLAIR', bet_first=False, num_threads=-1,
                 device='gpu', postproc=False):
        super().__init__(template_fpath, tmpdir, bet_modality=bet_modality,
                         bet_first=bet_first, num_threads=num_threads,
                         device=device)

        self.weights_fpath = weights_fpath

        self.postproc = postproc

    def _bet(self, modality_fpath):
        s_time = time()
        _, brain_mask_fpath = our_bet(
            modality_fpath,
            os.path.join(self.tmpdir, 'brain_'),
            self.weights_fpath,
            self.device,
            self.postproc
        )
        f_time = time()

        self.bet_cost_hist.append(f_time - s_time)

        return brain_mask_fpath
