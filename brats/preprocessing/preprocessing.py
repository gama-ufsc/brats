from abc import ABC, abstractmethod
import os

from typing import Tuple

import nibabel as nib

from .hdbet_wrapper import hd_bet
from .brainmage_wrapper import brainmage
from .nipype_wrappers import ants_registration, ants_transformation, fsl_bet, fsl_applymask


class Preprocessor(ABC):
    """Preprocessing module of BraTS pipeline.

    Aligns the FLAIR and T1 modalities to a T1 template using ANTs functions.
    Also performs brain extraction (BET defined by child).
    """
    def __init__(self, template_fpath: str, tmpdir: str, bet_modality='FLAIR',
                 bet_first=False, num_threads=-1):
        assert os.path.exists(template_fpath), (
            '`template` must be a valid path to the template image'
        )
        self.template_fpath = template_fpath

        self.num_threads = num_threads

        self.bet_modality = bet_modality.lower()

        self.bet_first = bet_first

        # create directory for temporary files
        os.makedirs(tmpdir, exist_ok=True)
        self.tmpdir = tmpdir

    def registration(self, flair_fpath: str, t1_fpath: str):
        transforms = list()

        # flair to t1 (within-subject) registration
        wsr_transform_fpath, _ = ants_registration(
            t1_fpath,
            flair_fpath,
            os.path.join(self.tmpdir, 'wsr_transform_'),
            self.num_threads,
        )
        transforms.insert(0, wsr_transform_fpath)

        # t1 (subject) to template registration
        str_transform_fpath, _ = ants_registration(
            self.template_fpath,
            t1_fpath,
            os.path.join(self.tmpdir, 'str_transform_'),
            self.num_threads,
        )
        transforms.insert(0, str_transform_fpath)

        # apply transformations to flair
        flair_at_template_fpath = ants_transformation(
            flair_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 'flair_template_'),
            self.num_threads,
        )

        # apply transformations to t1
        t1_at_template_fpath = ants_transformation(
            t1_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 't1_template_'),
            self.num_threads
        )

        return flair_at_template_fpath, t1_at_template_fpath, transforms

    @abstractmethod
    def _bet(self, modality_fpath) -> Tuple[str, str]:
        """ BETs `modality_fpath`, returning brain image and mask.
        """

    def bet_transform_apply(self, src_fpath, dst_fpath, transforms):
        """ BETs `src_fpath`, transforms the masks and applies in `dst_fpath`.
        """
        _, raw_brain_mask_fpath = self._bet(src_fpath)

        # transform mask to template space
        brain_mask_fpath = ants_transformation(
            raw_brain_mask_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 'mask_template_'),
            self.num_threads,
        )

        # apply mask
        brain_fpath = fsl_applymask(
            dst_fpath,
            brain_mask_fpath,
            os.path.join(self.tmpdir, 'brain_')
        )

        return brain_fpath, brain_mask_fpath

    def bet(self, flair_fpath, t1_fpath, flair_at_template_fpath,
            t1_at_template_fpath, transforms):
        if self.bet_modality == 't1':
            if self.bet_first:
                t1_brain_fpath, brain_mask_fpath = self.bet_transform_apply(
                    t1_fpath, t1_at_template_fpath, transforms
                )
            else:
                t1_brain_fpath, brain_mask_fpath = self._bet(t1_at_template_fpath)

            flair_brain_fpath = fsl_applymask(
                flair_at_template_fpath,
                brain_mask_fpath,
                os.path.join(self.tmpdir, 'brain_')
            )
        elif self.bet_modality == 'flair':
            if self.bet_first:
                flair_brain_fpath, brain_mask_fpath = self.bet_transform_apply(
                    flair_fpath, flair_at_template_fpath, transforms
                )
            else:
                flair_brain_fpath, brain_mask_fpath = self._bet(flair_at_template_fpath)

            t1_brain_fpath = fsl_applymask(
                t1_at_template_fpath,
                brain_mask_fpath,
                os.path.join(self.tmpdir, 'brain_')
            )
        elif self.bet_modality == 'all':
            if self.bet_first:
                flair_brain_fpath, _ = self.bet_transform_apply(
                    flair_fpath, flair_at_template_fpath, transforms
                )

                t1_brain_fpath, _ = self.bet_transform_apply(
                    t1_fpath, t1_at_template_fpath, transforms
                )
            else:
                flair_brain_fpath, _ = self._bet(flair_at_template_fpath)
                t1_brain_fpath, _ = self._bet(t1_at_template_fpath)

        return flair_brain_fpath, t1_brain_fpath

    def run(self, flair_fpath: str, t1_fpath: str) -> nib.Nifti1Image:
        """Run preprocessing pipeline.

        Args:
            flair_fpath: path to nifti image containing the FLAIR modality of
            the subject. Main object of the preprocessing operations.
            t1_fpath: Path to nifti image containing the T1 modality of the
            subject. Used to align the subject to the template.

        Returns:
            flair_image: Preprocessed and loaded `flair_image`.
            reverse_transform: Path to matrices that reverse the raw to
            preprocessed transformation.
        """
        (
            flair_at_template_fpath,
            t1_at_template_fpath,
            transforms
        ) = self.registration(flair_fpath, t1_fpath)

        flair_brain_fpath, t1_brain_fpath = self.bet(
            flair_fpath,
            t1_fpath,
            flair_at_template_fpath,
            t1_at_template_fpath,
            transforms
        )

        # load images
        flair_image = nib.load(flair_brain_fpath)
        t1_image = nib.load(t1_brain_fpath)

        return flair_image, t1_image, transforms

    def transform_prediction(self, pred: nib.Nifti1Image,
                             reverse_transforms: list, original_image: str
                            ) -> nib.Nifti1Image:
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
            reverse_transforms,
            os.path.join(self.tmpdir, 'transf_pred_'),
            self.num_threads,
            interpolation='NearestNeighbor',
            reverse=True,
        )

        return nib.load(transformed_pred_image)

class PreprocessorFSL(Preprocessor):
    """Preprocessing module of BraTS pipeline using FSL's BET.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.
    """
    def __init__(self, template_fpath: str, tmpdir: str, fast_bet=True,
                 bet_modality='FLAIR', bet_first=False, num_threads=-1):
        super().__init__(template_fpath, tmpdir, bet_modality=bet_modality,
                         bet_first=bet_first, num_threads=num_threads)

        self.fast_bet = fast_bet

    def _bet(self, modality_fpath):
        # extract brain
        brain_fpath, brain_mask_fpath = fsl_bet(
            modality_fpath,
            os.path.join(self.tmpdir, 'brain_'),
            fast=self.fast_bet
        )

        return brain_fpath, brain_mask_fpath

class PreprocessorHDBET(Preprocessor):
    """Preprocessing module of BraTS pipeline using HD-BET.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.
    """
    def __init__(self, template_fpath: str, tmpdir: str, bet_modality='FLAIR',
                 bet_first=False, num_threads=-1, **hdbet_kwargs):
        super().__init__(template_fpath, tmpdir, bet_modality=bet_modality,
                         bet_first=bet_first, num_threads=num_threads)

        self.hdbet_kwargs = hdbet_kwargs

    def _bet(self, modality_fpath):
        brain_fpath, brain_mask_fpath = hd_bet(
            modality_fpath,
            os.path.join(
                self.tmpdir,
                os.path.basename(modality_fpath).replace('.nii', '') \
                                                .replace('.gz', '')),
            **self.hdbet_kwargs
        )

        return brain_fpath, brain_mask_fpath

class PreprocessorBrainMaGe(Preprocessor):
    """Preprocessing module of BraTS pipeline using BrainMaGe.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.
    """
    def __init__(self, template_fpath: str, tmpdir: str, bet_modality='FLAIR',
                 bet_first=False, num_threads=-1, device='gpu'):
        super().__init__(template_fpath, tmpdir, bet_modality=bet_modality,
                         bet_first=bet_first, num_threads=num_threads)

        self.device = device

    def _bet(self, modality_fpath):
        brain_mask_fpath = os.path.join(
            self.tmpdir,
            'mask_' + os.path.basename(modality_fpath)
        )
        brain_mask_fpath = brainmage(modality_fpath, brain_mask_fpath,
                                     device=self.device)

        brain_fpath = fsl_applymask(
            modality_fpath,
            brain_mask_fpath,
            os.path.join(self.tmpdir, 'brain_')
        )

        return brain_fpath, brain_mask_fpath
