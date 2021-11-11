import importlib
import os
import shutil
import sys

from pathlib import Path
from warnings import warn

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from joblib import Parallel, delayed
from medpy.metric import hd95
from nibabel.viewers import OrthoSlicer3D
from nipype.interfaces.dcm2nii import Dcm2nii, Dcm2niix
from scipy.ndimage.morphology import distance_transform_edt as edt
from skimage.segmentation import find_boundaries


def get_orthoslicer(img, pos=(0, 0, 0), label=None, clipping=None):
    img_data = img.get_fdata()

    if label is not None:
        img_data = (img_data >= label).astype(int)

    v = OrthoSlicer3D(img_data, title=str(img.shape), affine=img.affine)

    if clipping is not None:
        bot_clip = np.quantile(img_data, clipping)
        top_clip = np.quantile(img_data, 1-clipping)
        v.clim = np.array([bot_clip, top_clip])
    else:
        unique_classes = np.unique(img_data)
        if len(unique_classes) < 50:  # no seg with more than 50 classes, right?
            # it seems that nibabel clips top 1% automatically
            v.clim = np.array([np.min(unique_classes), np.max(unique_classes)])

    v.set_position(*pos)

    return v


def _load_image_if_path(image):
    if isinstance(image, str) or isinstance(image, Path):
        image = Path(image).resolve()

        assert image.exists(), f'{image} does not exist'

        return nib.load(image)
    else:
        return image


def _bounding_box_from_overlay_data(overlay_data):
    overlay_ixs = np.nonzero(overlay_data)

    bb = np.zeros(overlay_data.shape)

    if not len(overlay_ixs[0]) * len(overlay_ixs[1]) == 0:
        y_min = overlay_ixs[0].min()
        y_max = overlay_ixs[0].max()
        x_min = overlay_ixs[1].min()
        x_max = overlay_ixs[1].max()

        bb[y_min, x_min:x_max] = 1  # bottom edge
        bb[y_max, x_min:x_max] = 1  # top edge
        bb[y_min:y_max, x_min] = 1  # left edge
        bb[y_min:y_max, x_max] = 1  # right edge

    return bb

def plot_img_overlay(image, overlay=None, overlay_label=None, alpha=0.25, ax=None):
    ol_colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
    ol_cmap = mcolors.LinearSegmentedColormap.from_list('ol_cmap', ol_colors, N=2)

    image_ = _load_image_if_path(image)
    if isinstance(image_, nib.Nifti1Image):
        image_ = image_.get_fdata()

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    im = ax.imshow(image_[:,:,0], cmap='gray', vmin=image_.min(), vmax=image_.max())

    im_pred = None
    if overlay is not None:
        overlay_ = _load_image_if_path(overlay)
        if isinstance(overlay_, nib.Nifti1Image):
            overlay_ = overlay_.get_fdata()

        if overlay_label is not None:
            overlay_ = (overlay_ >= overlay_label).astype(int)

        im_pred = ax.imshow(overlay_[:,:,0], alpha=alpha, cmap=ol_cmap, vmin=0, vmax=1)

    def update(layer):
        im.set_data(image_[:,:,layer])
        if im_pred is not None:
            im_pred.set_data(overlay_[:,:,layer])
        ax.figure.canvas.draw_idle()

    if isinstance(image, str) or isinstance(image, Path):
        ax.set_title(Path(image).name)

    return update, ax

def show_mri(image, overlay=None, overlay_label=None, pos=(-120, 120, 75),
             plot_bounding_box=False, clipping=None, alpha=0.25):
    ol_colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
    ol_cmap = mcolors.LinearSegmentedColormap.from_list('ol_cmap', ol_colors, N=2)

    image_ = _load_image_if_path(image)

    ortho_image = get_orthoslicer(image_, pos, clipping=clipping)

    if overlay is not None:
        overlay_ = _load_image_if_path(overlay)

        ortho_overlay = get_orthoslicer(overlay_, pos, label=overlay_label)

        # get generated figures for each
        image_fig = ortho_image.figs[0]
        overlay_fig = ortho_overlay.figs[0]

        # set aspect ratio of the image based on voxel size
        zooms = overlay_.header.get_zooms()
        aspect = [
            zooms[2] / zooms[0],
            zooms[2] / zooms[1],
            zooms[0] / zooms[1],
        ]

        for i in range(3):
            # get overlay data as plotted
            overlay_data = overlay_fig.axes[i].get_images()[0].get_array().data

            if plot_bounding_box:
                bb = _bounding_box_from_overlay_data(overlay_data)

                # plot bounding box on top of image
                image_fig.axes[i].imshow(bb, cmap=ol_cmap, aspect=aspect[i])
            else:
                # plot overlay data plotted on top of image
                image_fig.axes[i].imshow(overlay_data > 0, cmap=ol_cmap,
                                         alpha=alpha, aspect=aspect[i])

        ortho_overlay.close()

    ortho_image.show()


def _hd_distance(p: np.ndarray, l: np.ndarray, c: float = None,
                do_find_boundaries=True) -> np.ndarray:
    """Adapted from https://github.com/SilmarilBearer/HausdorffLoss/"""
    warn('LEGACY FUNCTION')
    if c is None:
        _p = (p > 0).astype(int)
        _l = (l > 0).astype(int)
    else:
        _p = (p >= c).astype(int)
        _l = (l >= c).astype(int)

    if np.count_nonzero(_p) == 0 or np.count_nonzero(_l) == 0:
        return float('inf')

    if do_find_boundaries:
        _p = find_boundaries(_p, mode='inner', background=0).astype(int)
        _l = find_boundaries(_l, mode='inner', background=0).astype(int)

    indexes_p = np.nonzero(_p)
    indexes_l = np.nonzero(_l)
    distances_l = edt(np.logical_not(_l))
    distances_p = edt(np.logical_not(_p))

    distances = np.concatenate((distances_l[indexes_p], distances_p[indexes_l]))

    return np.percentile(distances, 95)

def hd_distance(p: np.ndarray, l: np.ndarray, c: float = None) -> np.ndarray:
    if c is None:
        _p = (p > 0).astype(int)
        _l = (l > 0).astype(int)
    else:
        _p = (p >= c).astype(int)
        _l = (l >= c).astype(int)

    if np.all(_l == 0):
        if np.all(_p == 0):
            return 0
        else:
            return np.sqrt(np.sum(np.power(_l.shape, 2)))
    elif np.all(_p == 0):
        return np.sqrt(np.sum(np.power(_l.shape, 2)))
    else:
        return hd95(_p, _l, voxelspacing=1)

def compute_all_hausdorffs(preds_fpaths, labels_fpaths):
    scores = [list() for _ in  range(3)]

    for pred_fpath, label_fpath in zip(preds_fpaths, labels_fpaths):
        pred = nib.load(pred_fpath).get_fdata()
        label = nib.load(label_fpath).get_fdata()

        for c in range(1,3+1):
            scores[c-1].append(hd_distance(pred, label, c))

    # transpose
    scores = list(map(tuple, zip(*scores)))

    return scores

def dice(p: np.ndarray, l: np.ndarray, c: float = None):
    """ If class `c` is not provided, computes foreground dice (> 0).
    """
    if c is None:
        _p = (p > 0).astype(int)
        _l = (l > 0).astype(int)
    else:
        _p = (p >= c).astype(int)
        _l = (l >= c).astype(int)

    inter = (_p * _l).sum()
    union = _p.sum() + _l.sum()

    d = 2 * inter / union

    if np.isnan(d):
        return 0
    else:
        return d

def compute_foreground_dices(preds_fpaths, labels_fpaths):
    f_dices = list()
    for pred_fpath, label_fpath in zip(preds_fpaths, labels_fpaths):
        pred = nib.load(pred_fpath).get_fdata()
        label = nib.load(label_fpath).get_fdata()

        f_dices.append(dice(pred, label))

    return f_dices

def compute_all_dices(preds_fpaths, labels_fpaths, n_jobs=-1):
    scores = [list() for _ in  range(3)]

    if 'tqdm' in sys.modules:
        tqdm_ = importlib.import_module('tqdm')
        if 'tqdm.notebook' in sys.modules:
            tqdm = tqdm_.notebook.tqdm
        else:
            tqdm = tqdm_.tqdm
    else:
        tqdm = lambda x: x

    for pred_fpath, label_fpath in tqdm(list(zip(preds_fpaths, labels_fpaths))):
        pred = nib.load(pred_fpath).get_fdata()
        label = nib.load(label_fpath).get_fdata()

        scores_ = list()
        for c in range(1,3+1):
            scores_.append(delayed(dice)(pred, label, c))

        scores_ = Parallel(n_jobs=n_jobs)(scores_)
        for c in range(1,3+1):
            scores[c-1].append(scores_[c-1])

    # transpose
    scores = list(map(tuple, zip(*scores)))

    return scores

def compute_all_scores(preds_fpaths, labels_fpaths, n_jobs=-1):
    scores = {'Dice': list(), 'HD95': list()}

    if 'tqdm' in sys.modules:
        tqdm_ = importlib.import_module('tqdm')
        if 'tqdm.notebook' in sys.modules:
            tqdm = tqdm_.notebook.tqdm
        else:
            tqdm = tqdm_.tqdm
    else:
        tqdm = lambda x: x

    for pred_fpath, label_fpath in tqdm(list(zip(preds_fpaths, labels_fpaths))):
        pred = nib.load(pred_fpath).get_fdata()
        label = nib.load(label_fpath).get_fdata()

        dices = list()
        hds = list()
        for c in range(1,3+1):
            dices.append(delayed(dice)(pred, label, c))
            hds.append(delayed(hd_distance)(pred, label, c))

        res = Parallel(n_jobs=n_jobs)(dices + hds)
        dices = res[:3]
        hds = res[3:]

        scores['Dice'].append(dices)
        scores['HD95'].append(hds)

    return scores

def dcm2nifti(dcm_fpaths, tmpdir):
    conv = Dcm2niix()

    conv.inputs.source_dir = str(dcm_fpaths)
    conv.inputs.output_dir = str(tmpdir)

    res = conv.run()

    def _move(src):
        dst = src.parent / src.name.replace('(', '').replace(')', '').replace(',', '')

        shutil.move(src, dst)

        return dst

    # fix bad characters in filepath
    if isinstance(res.outputs.converted_files, list):
        fpaths = list()
        for fpath in res.outputs.converted_files:
            src = Path(fpath)
            dst = _move(src)

            fpaths.append(dst)
        return fpaths
    else:
        src = Path(res.outputs.converted_files)

        return _move(src)
