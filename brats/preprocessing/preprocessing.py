import os

import nibabel as nib

from .hdbet_wrapper import hd_bet
from .brainmage_wrapper import brainmage
from .nipype_wrappers import ants_registration, ants_transformation, fsl_bet, fsl_applymask


class Preprocessor():
    """Preprocessing module of BraTS pipeline.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.

    This version performs registration and transformation first (FLAIR to T1 to
    template) and brain extraction last. Registration and transformation are
    using ANTS; brain extraction uses FSL's BET.
    """
    def __init__(self, template_fpath: str, tmpdir: str, fast_bet: bool = True,
                 preprocess_t1=True, bet_flair=True, num_threads=-1):
        assert os.path.exists(template_fpath), (
            '`template` must be a valid path to the template image'
        )
        self.template_fpath = template_fpath

        self.fast_bet = fast_bet

        self.num_threads = num_threads

        self.preprocess_t1 = preprocess_t1

        self.bet_flair = bet_flair

        # create directory for temporary files
        os.makedirs(tmpdir, exist_ok=True)
        self.tmpdir = tmpdir

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
        transforms = list()
        reverse_transforms = list()

        # flair to t1 (within-subject) registration
        wsr_transform_fpath, wsr_reverse_transform_fpath = ants_registration(
            t1_fpath,
            flair_fpath,
            os.path.join(self.tmpdir, 'wsr_transform_'),
            self.num_threads,
        )
        transforms.insert(0, wsr_transform_fpath)
        reverse_transforms.append(wsr_reverse_transform_fpath)

        # t1 (subject) to template registration
        str_transform_fpath, str_reverse_transform_fpath = ants_registration(
            self.template_fpath,
            t1_fpath,
            os.path.join(self.tmpdir, 'str_transform_'),
            self.num_threads,
        )
        transforms.insert(0, str_transform_fpath)
        reverse_transforms.append(str_reverse_transform_fpath)

        # apply transformations to flair
        flair_at_template_fpath = ants_transformation(
            flair_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 'flair_template_'),
            self.num_threads
        )

        if self.bet_flair:
            # extract brain from flair
            flair_brain_fpath, brain_mask_fpath = fsl_bet(
                flair_at_template_fpath,
                os.path.join(self.tmpdir, 'flair_brain_'),
                fast=self.fast_bet
            )

            # load image
            flair_image = nib.load(flair_brain_fpath)

            if self.preprocess_t1:
                # apply transformations to t1
                t1_at_template_fpath = ants_transformation(
                    t1_fpath,
                    self.template_fpath,
                    transforms,
                    os.path.join(self.tmpdir, 't1_template_'),
                    self.num_threads
                )

                # extract brain from t1
                t1_brain_fpath = fsl_applymask(
                    t1_at_template_fpath,
                    brain_mask_fpath,
                    os.path.join(self.tmpdir, 'brain_')
                )

                # load image
                t1_image = nib.load(t1_brain_fpath)

                return flair_image, t1_image, reverse_transforms[::-1]
        else:
            # apply transformations to t1
            t1_at_template_fpath = ants_transformation(
                t1_fpath,
                self.template_fpath,
                transforms,
                os.path.join(self.tmpdir, 't1_template_'),
                self.num_threads
            )

            # extract brain from t1
            t1_brain_fpath, brain_mask_fpath = fsl_bet(
                t1_at_template_fpath,
                os.path.join(self.tmpdir, 't1_brain_'),
                fast=self.fast_bet
            )

            # extract brain from flair
            flair_brain_fpath = fsl_applymask(
                flair_at_template_fpath,
                brain_mask_fpath,
                os.path.join(self.tmpdir, 'brain_')
            )

            # load image
            flair_image = nib.load(flair_brain_fpath)

            if self.preprocess_t1:
                # load image
                t1_image = nib.load(t1_brain_fpath)

                return flair_image, t1_image, reverse_transforms[::-1]
            else:
                return flair_image, reverse_transforms[::-1]

    def transform_prediction(
        self,
        pred: nib.Nifti1Image,
        reverse_transforms: list,
        original_image: str
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


class PreprocessorHDBET(Preprocessor):
    """Preprocessing module of BraTS pipeline.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.

    This version performs registration and transformation first (FLAIR to T1 to
    template) and brain extraction last. Registration and transformation are
    using ANTS; brain extraction uses FSL's BET.
    """
    def __init__(self, template_fpath: str, tmpdir: str, bet_all=True,
                 bet_first=False, num_threads=-1, **hdbet_kwargs):
        super().__init__(template_fpath, tmpdir, fast_bet=False,
                         preprocess_t1=True, num_threads=num_threads)
        self.bet_all = bet_all
        self.bet_first = bet_first
        self.hdbet_kwargs = hdbet_kwargs

    def run(self, flair_fpath: str, t1_fpath: str) -> nib.Nifti1Image:
        """Run preprocessing pipeline.

        Args:
            flair_fpath: path to nifti image containing the FLAIR modality of
            the subject. Main object of the preprocessing operations.
            t1_fpath: Path to nifti image containing the T1 modality of the
            subject. Used to align the subject to the template.

        Returns:
            flair_image: Preprocessed and loaded `flair_image`.
            t1_image: Preprocessed and loaded `t1_image`.
            reverse_transform: Path to matrices that reverse the raw to
            preprocessed transformation.
        """
        transforms = list()
        reverse_transforms = list()

        # flair to t1 (within-subject) registration
        wsr_transform_fpath, wsr_reverse_transform_fpath = ants_registration(
            t1_fpath,
            flair_fpath,
            os.path.join(self.tmpdir, 'wsr_transform_'),
            self.num_threads,
        )
        transforms.insert(0, wsr_transform_fpath)
        reverse_transforms.append(wsr_reverse_transform_fpath)

        # t1 (subject) to template registration
        str_transform_fpath, str_reverse_transform_fpath = ants_registration(
            self.template_fpath,
            t1_fpath,
            os.path.join(self.tmpdir, 'str_transform_'),
            self.num_threads,
        )
        transforms.insert(0, str_transform_fpath)
        reverse_transforms.append(str_reverse_transform_fpath)

        # apply transformations to flair
        flair_at_template_fpath = ants_transformation(
            flair_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 'flair_template_'),
            self.num_threads
        )
        # apply transformations to t1
        t1_at_template_fpath = ants_transformation(
            t1_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 't1_template_'),
            self.num_threads
        )

        if not self.bet_first:
            t1_brain_fpath, brain_mask_fpath = hd_bet(
                t1_at_template_fpath,
                os.path.join(self.tmpdir,
                             os.path.basename(t1_fpath).replace('.nii', '').replace('.gz', '')),
                **self.hdbet_kwargs
            )
        else:
            _, raw_t1_mask_fpath = hd_bet(
                t1_fpath,
                os.path.join(self.tmpdir,
                             os.path.basename(t1_fpath).replace('.nii', '').replace('.gz', '')),
                **self.hdbet_kwargs
            )

            # transform mask to template space
            brain_mask_fpath = ants_transformation(
                raw_t1_mask_fpath,
                self.template_fpath,
                transforms,
                os.path.join(self.tmpdir, 'mask_template_'),
                self.num_threads
            )

            # apply mask to t1 at template
            t1_brain_fpath = fsl_applymask(
                t1_at_template_fpath,
                brain_mask_fpath,
                os.path.join(self.tmpdir, 'brain_')
            )

        if self.bet_all:
            if not self.bet_first:
                flair_brain_fpath, _ = hd_bet(
                    flair_at_template_fpath,
                    os.path.join(self.tmpdir,
                                os.path.basename(flair_fpath).replace('.nii', '').replace('.gz', '')),
                    **self.hdbet_kwargs
                )
            else:
                _, raw_flair_mask_fpath = hd_bet(
                    flair_fpath,
                    os.path.join(self.tmpdir,
                                os.path.basename(flair_fpath).replace('.nii', '').replace('.gz', '')),
                    **self.hdbet_kwargs
                )

                # transform mask to template space
                brain_mask_fpath = ants_transformation(
                    raw_flair_mask_fpath,
                    self.template_fpath,
                    transforms,
                    os.path.join(self.tmpdir, 'mask_template_'),
                    self.num_threads
                )

                # apply mask to flair at template
                flair_brain_fpath = fsl_applymask(
                    flair_at_template_fpath,
                    brain_mask_fpath,
                    os.path.join(self.tmpdir, 'brain_')
                )
        else:
            flair_brain_fpath = fsl_applymask(
                flair_at_template_fpath,
                brain_mask_fpath,
                os.path.join(self.tmpdir, 'brain_')
            )

        flair_image = nib.load(flair_brain_fpath)
        t1_image = nib.load(t1_brain_fpath)

        return flair_image, t1_image, reverse_transforms[::-1]

class PreprocessorBrainMaGe(Preprocessor):
    """Preprocessing module of BraTS pipeline.

    Aligns the FLAIR and T1 modalities to a T1 template. Also performs brain
    extraction.

    This version performs registration and transformation first (FLAIR to T1 to
    template) and brain extraction last. Registration and transformation are
    using ANTS; brain extraction uses BrainMaGe.
    """
    def __init__(self, template_fpath: str, tmpdir: str, device='gpu',
                 num_threads=-1):
        super().__init__(template_fpath, tmpdir, fast_bet=False,
                         preprocess_t1=True, num_threads=num_threads)
        self.device = device

    def run(self, flair_fpath: str, t1_fpath: str) -> nib.Nifti1Image:
        """Run preprocessing pipeline.

        Args:
            flair_fpath: path to nifti image containing the FLAIR modality of
            the subject. Main object of the preprocessing operations.
            t1_fpath: Path to nifti image containing the T1 modality of the
            subject. Used to align the subject to the template.

        Returns:
            flair_image: Preprocessed and loaded `flair_image`.
            t1_image: Preprocessed and loaded `t1_image`.
            reverse_transform: Path to matrices that reverse the raw to
            preprocessed transformation.
        """
        transforms = list()
        reverse_transforms = list()

        # flair to t1 (within-subject) registration
        wsr_transform_fpath, wsr_reverse_transform_fpath = ants_registration(
            t1_fpath,
            flair_fpath,
            os.path.join(self.tmpdir, 'wsr_transform_'),
            self.num_threads,
        )
        transforms.insert(0, wsr_transform_fpath)
        reverse_transforms.append(wsr_reverse_transform_fpath)

        # t1 (subject) to template registration
        str_transform_fpath, str_reverse_transform_fpath = ants_registration(
            self.template_fpath,
            t1_fpath,
            os.path.join(self.tmpdir, 'str_transform_'),
            self.num_threads,
        )
        transforms.insert(0, str_transform_fpath)
        reverse_transforms.append(str_reverse_transform_fpath)

        # apply transformations to flair
        flair_at_template_fpath = ants_transformation(
            flair_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 'flair_template_'),
            self.num_threads
        )
        # apply transformations to t1
        t1_at_template_fpath = ants_transformation(
            t1_fpath,
            self.template_fpath,
            transforms,
            os.path.join(self.tmpdir, 't1_template_'),
            self.num_threads
        )

        brain_mask_fpath = os.path.join(
            self.tmpdir,
            'mask_' + os.path.basename(t1_at_template_fpath)
        )
        brain_mask_fpath = brainmage(t1_at_template_fpath, brain_mask_fpath,
                                     device=self.device)

        t1_brain_fpath = fsl_applymask(
            t1_at_template_fpath,
            brain_mask_fpath,
            os.path.join(self.tmpdir, 'brain_')
        )

        flair_brain_fpath = fsl_applymask(
            flair_at_template_fpath,
            brain_mask_fpath,
            os.path.join(self.tmpdir, 'brain_')
        )

        flair_image = nib.load(flair_brain_fpath)
        t1_image = nib.load(t1_brain_fpath)

        return flair_image, t1_image, reverse_transforms[::-1]
