import os

from pathlib import Path

import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np

from nibabel.viewers import OrthoSlicer3D


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
