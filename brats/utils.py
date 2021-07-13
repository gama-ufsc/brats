import os

from pathlib import Path

import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np

from nibabel.viewers import OrthoSlicer3D
from nipype.interfaces.dcm2nii import Dcm2nii, Dcm2niix
from scipy.ndimage.morphology import distance_transform_edt as edt


def get_orthoslicer(img, pos=(0, 0, 0), clipping=0.0):
    img_data = img.get_fdata()

    # clipping
    bot_clip = np.quantile(img_data, clipping)
    top_clip = np.quantile(img_data, 1-clipping)
    img_data[img_data < bot_clip] = bot_clip
    img_data[img_data > top_clip] = top_clip

    v = OrthoSlicer3D(img_data, title=str(img.shape), affine=img.affine)
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


def show_mri(image, overlay=None, pos=(0, 0, 0), plot_bounding_box=False,
             clipping=0.0, alpha=0.25):
    ol_colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
    ol_cmap = mcolors.LinearSegmentedColormap.from_list('ol_cmap', ol_colors, N=2)

    image_ = _load_image_if_path(image)

    ortho_image = get_orthoslicer(image_, pos, clipping=clipping)

    if overlay is not None:
        overlay_ = _load_image_if_path(overlay)

        ortho_overlay = get_orthoslicer(overlay_, pos)

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

    return 2 * inter / union

def compute_foreground_dices(preds_fpaths, labels_fpaths):
    f_dices = list()
    for pred_fpath, label_fpath in zip(preds_fpaths, labels_fpaths):
        pred = nib.load(pred_fpath).get_fdata()
        label = nib.load(label_fpath).get_fdata()

        f_dices.append(dice(pred, label))

    return f_dices

def compute_all_dices(preds_fpaths, labels_fpaths):
    scores = [list() for _ in  range(3)]

    for pred_fpath, label_fpath in zip(preds_fpaths, labels_fpaths):
        pred = nib.load(pred_fpath).get_fdata()
        label = nib.load(label_fpath).get_fdata()

        for c in range(1,3+1):
            scores[c-1].append(dice(pred, label, c))

    # transpose
    scores = list(map(tuple, zip(*scores)))

    return scores

def dcm2nifti(dcm_fpaths, tmpdir):
    conv = Dcm2niix()

    conv.inputs.source_dir = str(dcm_fpaths)
    conv.inputs.output_dir = str(tmpdir)

    res = conv.run()

    return res.outputs.converted_files

def hd_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """From https://github.com/SilmarilBearer/HausdorffLoss/"""
    if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
        return np.array([np.Inf])

    indexes = np.nonzero(x)
    distances = edt(np.logical_not(y))

    # return np.array(np.max(distances[indexes]))
    return np.array(np.quantile(distances[indexes], 0.95))
