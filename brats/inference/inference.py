import math

from typing import List

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label
from torch.nn import functional as F


class UNet2D(nn.Module):
    """2D U-Net model.

    This code is based on "U-Net: Convolutional Networks for Biomedical Image
    Segmentation" by Ronneberger et al.
    """
    def __init__(self, in_channels, out_channels, init_depth):
        super().__init__()

        assert in_channels >= 1, "in_channels must be greater than 0"
        assert out_channels >= 1, "out_channels must be greater than 0"
        assert init_depth >= 1, "init_depth must be greater than 0"

        # Encoder part
        self.Encoder_1 = self.layers_block(in_channels, init_depth)
        self.Encoder_MaxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Encoder_2 = self.layers_block(1*init_depth, 2*init_depth)
        self.Encoder_MaxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Encoder_3 = self.layers_block(2*init_depth, 4*init_depth)
        self.Encoder_MaxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Encoder_4 = self.layers_block(4*init_depth, 8*init_depth)
        self.Encoder_MaxPool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Encoder_5 = self.layers_block(8*init_depth, 16*init_depth)
        self.Encoder_MaxPool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck part
        self.Encoder_6 = self.layers_block(16*init_depth, 16*init_depth)

        # Decoder part
        self.Decoder_UpSampling_1 = nn.ConvTranspose2d(
            in_channels=16*init_depth,
            out_channels=16*init_depth,
            kernel_size=2,
            stride=2
        )
        self.Decoder_1 = self.layers_block(32*init_depth, 8*init_depth)
        self.Decoder_UpSampling_2 = nn.ConvTranspose2d(
            in_channels=8*init_depth,
            out_channels=8*init_depth,
            kernel_size=2,
            stride=2,
        )
        self.Decoder_2 = self.layers_block(16*init_depth, 4*init_depth)
        self.Decoder_UpSampling_3 = nn.ConvTranspose2d(
            in_channels=4*init_depth,
            out_channels=4*init_depth,
            kernel_size=2,
            stride=2,
        )
        self.Decoder_3 = self.layers_block(8*init_depth, 2*init_depth)
        self.Decoder_UpSampling_4 = nn.ConvTranspose2d(
            in_channels=2*init_depth,
            out_channels=2*init_depth,
            kernel_size=2,
            stride=2,
        )
        self.Decoder_4 = self.layers_block(4*init_depth, 1*init_depth)
        self.Decoder_UpSampling_5 = nn.ConvTranspose2d(
            in_channels=1*init_depth,
            out_channels=1*init_depth,
            kernel_size=2,
            stride=2,
        )
        self.Decoder_5 = self.layers_block(2*init_depth, 1*init_depth)

        # Output
        self.Output = nn.Conv2d(in_channels=init_depth,
                                out_channels=out_channels, kernel_size=1,
                                stride=1)

    @staticmethod
    def layers_block(in_channels, out_channels):
        """ Create a convolutional block for the U-Net implementation.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    @staticmethod
    def calculate_pad(x_1, x_2):
        padding_x = np.uint8(x_1.shape[2] - x_2.shape[2])
        padding_y = np.uint8(x_1.shape[3] - x_2.shape[3])

        first_dim_x = math.floor(padding_x/2)
        second_dim_x = padding_x - first_dim_x
        first_dim_y = math.floor(padding_y/2)
        second_dim_y = padding_y - first_dim_y

        return F.pad(x_2, (first_dim_y, second_dim_y, first_dim_x, second_dim_x))

    def forward(self, x):
        encoder_1 = self.Encoder_1(x)
        encoder_2 = self.Encoder_2(self.Encoder_MaxPool_1(encoder_1))
        encoder_3 = self.Encoder_3(self.Encoder_MaxPool_2(encoder_2))
        encoder_4 = self.Encoder_4(self.Encoder_MaxPool_3(encoder_3))
        encoder_5 = self.Encoder_5(self.Encoder_MaxPool_4(encoder_4))
        encoder_6 = self.Encoder_6(self.Encoder_MaxPool_5(encoder_5))
        encoder_6 = self.calculate_pad(encoder_5, self.Decoder_UpSampling_1(encoder_6))
        decoder_1 = self.Decoder_1(torch.cat([encoder_5, encoder_6], dim=1))
        decoder_1 = self.calculate_pad(encoder_4, self.Decoder_UpSampling_2(decoder_1))
        decoder_2 = self.Decoder_2(torch.cat([encoder_4, decoder_1], dim=1))
        decoder_2 = self.calculate_pad(encoder_3, self.Decoder_UpSampling_3(decoder_2))
        decoder_3 = self.Decoder_3(torch.cat([encoder_3, decoder_2], dim=1))
        decoder_3 = self.calculate_pad(encoder_2, self.Decoder_UpSampling_4(decoder_3))
        decoder_4 = self.Decoder_4(torch.cat([encoder_2, decoder_3], dim=1))
        decoder_4 = self.calculate_pad(encoder_1, self.Decoder_UpSampling_5(decoder_4))
        decoder_5 = self.Decoder_5(torch.cat([encoder_1, decoder_4], dim=1))

        output = self.Output(decoder_5)

        return output


class BraTSModel():
    def __init__(self, weights_fpath, in_channels=2, device='cuda:0') -> None:
        self.device = torch.device(device)

        self.load_model(weights_fpath, in_channels)
        self.in_channels = in_channels

    def load_model(self, model_fpath, in_channels=1):
        model = UNet2D(in_channels, 1, 32)

        model.load_state_dict(torch.load(model_fpath, map_location=self.device))

        model.eval()

        self._model = model

    def prepare_images(self, images: List[nib.Nifti1Image]) -> np.ndarray:
        """Get image ready for the model.
        """
        xs = [self.prepare_image(image) for image in images]

        x = np.concatenate(xs, axis=1)

        return x

    def prepare_image(self, image: nib.Nifti1Image) -> np.ndarray:
        """Get multiple images ready for the model as different channels.
        """
        x = image.get_fdata().copy()

        # reshape data
        x = x.transpose(2, 0, 1)
        x = np.flip(x, axis=2)

        # pad image
        x = x[:, 40:200, 30:222]  # we want it in (155, 160, 192) shape

        # normalization
        brain_mask = x != 0  # background won't be affected
        x[brain_mask] = (x[brain_mask] - x[brain_mask].mean()) \
                        / (x[brain_mask].std() + 1e-6)

        return np.expand_dims(x, axis=1)

    def inference(self, image: np.ndarray, confidence=False) -> np.ndarray:
        """Feed image to the model, returning the segmentation.
        """
        with torch.no_grad():
            # shape image into a model-friendly torch.Tensor
            x = torch.tensor(image.copy())
            x = x.to(self.device).float()

            pred = np.zeros((155, 160, 192))

            # iterate over the layers, one prediction for each
            for j in range(155):
                y_pred = torch.sigmoid(
                    self._model(x[j].unsqueeze(dim=0))
                )
                # get prediction back to cpu (if it ever left it)
                pred[j] = y_pred[0, 0].cpu().detach().numpy()

        if not confidence:
            # sigmoid prediction to binary
            pred = (pred >= 0.5).astype(int)

        return pred

    def prepare_pred(self, pred: np.ndarray, og_image: nib.Nifti1Image) -> nib.Nifti1Image:
        """Get prediction into NIfTI format based on `og_image`.
        """
        pred_data = np.zeros((155, 240, 240))

        # the opposite of `self.prepare_image`
        pred_data[:, 40:200, 30:222] = pred
        pred_data = np.flip(pred_data, axis=2)
        pred_data = pred_data.transpose(1, 2, 0)

        pred_image = nib.Nifti1Image(pred_data, affine=og_image.affine,
                                     header=og_image.header)

        return pred_image

    def component_filter(self, pred: np.ndarray, absolute_threshold: int = 0,
                         relative_threshold: float = 0.0) -> np.ndarray:
        """Filter components of `pred` smaller than both thresholds.

        Components are determined from the binary form of `pred` (`pred` > 0)
        using `skimage.measure.label` with connectivity 3. Then, components
        smaller than `absolute_threshold` OR `relative_threshold` * (size of
        the largest component in `pred`) are discarded.
        """
        pred_ = np.copy(pred)

        binary_pred = (pred > 0).astype(int)

        # get components
        pred_labeled = label(binary_pred, connectivity=3, background=0)

        components = np.unique(pred_labeled)
        components = np.array([c for c in components if c != 0])  # 0 = background

        # components' sizes in voxels
        components_sizes = np.array([np.sum(pred_labeled == c) for c in components])

        # only components larger than absolute AND relative
        thresh = max(relative_threshold * max(components_sizes),
                     absolute_threshold)
        big_enough_components = components[components_sizes >= thresh]

        # remove small components
        pred_[~np.isin(pred_labeled, big_enough_components)] = 0

        return pred_

    def remove_peripheral_components(self, pred: np.ndarray, x: np.ndarray,
                                     min_distance: float = 5) -> np.ndarray:
        """Remove peripheral components from `pred`.

        Components are considered peripheral if they are less than
        `min_distance` from the background of `x`
        """
        # group background voxels
        bg_labels = label(x == 0, connectivity=3)
        # then take the largest one
        bg_label = np.argmax(np.bincount(bg_labels.flatten())[1:]) + 1
        # everything else is foreground :)
        fg = bg_labels != bg_label

        # distance of every fg voxel to the neares bg voxel
        fg_dt = distance_transform_edt(fg)

        # get prediction components that are at least `min_distance` from bg
        pred_comps = label(pred)

        comps_min_distance = dict()
        for comp in np.unique(pred_comps):
            comps_min_distance[comp] = np.min(fg_dt[pred_comps == comp])
        comps_to_keep = [c for c, d in comps_min_distance.items() if d > min_distance]

        return pred * np.isin(pred_comps, comps_to_keep)

    def __call__(self, image: List[nib.Nifti1Image]) -> nib.Nifti1Image:
        """Run standard use of the model.
        """
        if type(image) == list:
            xs = self.prepare_images(image)

            pred = self.inference(xs)

            pred_image = self.prepare_pred(pred, image[0])
        else:
            x = self.prepare_image(image)

            pred = self.inference(x)

            pred_image = self.prepare_pred(pred, image)

        return pred_image
