import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from skimage.transform import resize

torch.manual_seed(0)

class UNet2D(nn.Module):
    """
    This code is based in this article - U-Net: Convolutional Networks for Biomedical Image Segmentation - https://arxiv.org/pdf/1505.04597.pdf
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
        self.Decoder_UpSampling_1 = nn.ConvTranspose2d(in_channels=16*init_depth, out_channels=16*init_depth, kernel_size=2, stride=2)
        self.Decoder_1 = self.layers_block(32*init_depth, 8*init_depth)
        self.Decoder_UpSampling_2 = nn.ConvTranspose2d(in_channels=8*init_depth, out_channels=8*init_depth, kernel_size=2, stride=2)
        self.Decoder_2 = self.layers_block(16*init_depth, 4*init_depth)
        self.Decoder_UpSampling_3 = nn.ConvTranspose2d(in_channels=4*init_depth, out_channels=4*init_depth, kernel_size=2, stride=2)
        self.Decoder_3 = self.layers_block(8*init_depth, 2*init_depth)
        self.Decoder_UpSampling_4 = nn.ConvTranspose2d(in_channels=2*init_depth, out_channels=2*init_depth, kernel_size=2, stride=2)
        self.Decoder_4 = self.layers_block(4*init_depth, 1*init_depth)
        self.Decoder_UpSampling_5 = nn.ConvTranspose2d(in_channels=1*init_depth, out_channels=1*init_depth, kernel_size=2, stride=2)
        self.Decoder_5 = self.layers_block(2*init_depth, 1*init_depth)
        # Output
        self.Output_1 = nn.Conv2d(
            in_channels=init_depth, out_channels=out_channels, kernel_size=1, stride=1)
        self.Output_2 = nn.Conv2d(
            in_channels=2*init_depth, out_channels=out_channels, kernel_size=1, stride=1)
        self.Output_3 = nn.Conv2d(
            in_channels=init_depth, out_channels=out_channels, kernel_size=1, stride=1)

        
    @staticmethod
    def layers_block(in_channels, out_channels):
        """
        This static method create a convolutional block useful for the U-Net implementation
        """
        return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(num_features=out_channels),
                             nn.ReLU(),
                             nn.Conv2d(
                                 in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
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
        output_2 = self.Output_2(decoder_3)
        
        decoder_3 = self.calculate_pad(encoder_2, self.Decoder_UpSampling_4(decoder_3))        
        decoder_4 = self.Decoder_4(torch.cat([encoder_2, decoder_3], dim=1))
        output_3 = self.Output_3(decoder_4)
        
        decoder_4 = self.calculate_pad(encoder_1, self.Decoder_UpSampling_5(decoder_4))
        decoder_5 = self.Decoder_5(torch.cat([encoder_1, decoder_4], dim=1))
        output_1 = self.Output_1(decoder_5)
        return output_1, output_2, output_3

    
def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class EncodingModule(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, dropout_p=0.3, leakiness=1e-2, conv_bias=True,
                 inst_norm_affine=True, lrelu_inplace=True):
        nn.Module.__init__(self)
        self.dropout_p = dropout_p
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.bn_1 = nn.InstanceNorm2d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, filter_size, 1, (filter_size - 1) // 2, bias=self.conv_bias)
        self.dropout = nn.Dropout2d(dropout_p)
        self.bn_2 = nn.InstanceNorm2d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, filter_size, 1, (filter_size - 1) // 2, bias=self.conv_bias)

    def forward(self, x):
        skip = x
        x = F.leaky_relu(self.bn_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.bn_2(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv2(x)
        x = x + skip
        return x


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bicubic', align_corners=True):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=self.conv_bias)
        self.bn_1 = nn.InstanceNorm2d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=self.conv_bias)
        self.bn_2 = nn.InstanceNorm2d(out_channels, affine=self.inst_norm_affine, track_running_stats=True)

    def forward(self, x):
        x = F.leaky_relu(self.bn_1(self.conv1(x)), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = F.leaky_relu(self.bn_2(self.conv2(x)), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        return x


class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.upsample = Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        self.upsample_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=self.conv_bias)
        self.bn = nn.InstanceNorm2d(out_channels, affine=self.inst_norm_affine, track_running_stats=True)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.upsample_conv(self.upsample(x))), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        return x


class DownsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.bn = nn.InstanceNorm2d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.downsample = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=self.conv_bias)

    def forward(self, x):
        x = F.leaky_relu(self.bn(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        b = self.downsample(x)
        return x, b


class Network(nn.Module):
    def __init__(self, num_classes=1, num_input_channels=1, base_filters=16, dropout_p=0.3,
                 final_nonlin=softmax_helper, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True, do_ds=True):
        super(Network, self).__init__()

        self.do_ds = do_ds
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.final_nonlin = final_nonlin
        self.init_conv = nn.Conv2d(num_input_channels, base_filters, 3, 1, 1, bias=self.conv_bias)

        self.context1 = EncodingModule(base_filters, base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down1 = DownsamplingModule(base_filters, base_filters * 2, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context2 = EncodingModule(2 * base_filters, 2 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down2 = DownsamplingModule(2 * base_filters, base_filters * 4, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context3 = EncodingModule(4 * base_filters, 4 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down3 = DownsamplingModule(4 * base_filters, base_filters * 8, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context4 = EncodingModule(8 * base_filters, 8 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down4 = DownsamplingModule(8 * base_filters, base_filters * 16, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context5 = EncodingModule(16 * base_filters, 16 * base_filters, 3, dropout_p, leakiness=1e-2,
                                       conv_bias=True, inst_norm_affine=True, lrelu_inplace=True)

        self.bn_after_context5 = nn.InstanceNorm2d(16 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.up1 = UpsamplingModule(16 * base_filters, 8 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc1 = LocalizationModule(16 * base_filters, 8 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.up2 = UpsamplingModule(8 * base_filters, 4 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc2 = LocalizationModule(8 * base_filters, 4 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.loc2_seg = nn.Conv2d(4 * base_filters, num_classes, 1, 1, 0, bias=False)
        self.up3 = UpsamplingModule(4 * base_filters, 2 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc3 = LocalizationModule(4 * base_filters, 2 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.loc3_seg = nn.Conv2d(2 * base_filters, num_classes, 1, 1, 0, bias=False)
        self.up4 = UpsamplingModule(2 * base_filters, 1 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.end_conv_1 = nn.Conv2d(2 * base_filters, 2 * base_filters, 3, 1, 1, bias=self.conv_bias)
        self.end_conv_1_bn = nn.InstanceNorm2d(2 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.end_conv_2 = nn.Conv2d(2 * base_filters, 2 * base_filters, 3, 1, 1, bias=self.conv_bias)
        self.end_conv_2_bn = nn.InstanceNorm2d(2 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.seg_layer = nn.Conv2d(2 * base_filters, num_classes, 1, 1, 0, bias=False)

    def forward(self, x):
        seg_outputs = []

        x = self.init_conv(x)
        x = self.context1(x)

        skip1, x = self.down1(x)
        x = self.context2(x)

        skip2, x = self.down2(x)
        x = self.context3(x)

        skip3, x = self.down3(x)
        x = self.context4(x)

        skip4, x = self.down4(x)
        x = self.context5(x)

        x = F.leaky_relu(self.bn_after_context5(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.up1(x)

        x = torch.cat((skip4, x), dim=1)
        x = self.loc1(x)
        x = self.up2(x)

        x = torch.cat((skip3, x), dim=1)
        x = self.loc2(x)
        loc2_seg = self.final_nonlin(self.loc2_seg(x))
        seg_outputs.append(loc2_seg)
        x = self.up3(x)

        x = torch.cat((skip2, x), dim=1)
        x = self.loc3(x)
        loc3_seg = self.final_nonlin(self.loc3_seg(x))
        seg_outputs.append(loc3_seg)
        x = self.up4(x)

        x = torch.cat((skip1, x), dim=1)
        x = F.leaky_relu(self.end_conv_1_bn(self.end_conv_1(x)), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = F.leaky_relu(self.end_conv_2_bn(self.end_conv_2(x)), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = self.seg_layer(x)
        seg_outputs.append(x)

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]

def predict(x, x_or):

    UNet =  Network(base_filters=24).to("cuda:0")
    UNet.load_state_dict(torch.load("./hd_bet/unet_2d_hp_tests_with_data_augmentation.pth"))
    UNet.to("cuda:0")

    prediction = np.zeros(x.shape)

    x = torch.tensor(x)

    # UNet.eval()

    with torch.no_grad():
        for i in range(x.shape[0]):
            x_s = x[i]
            x_s = x_s.to("cuda:0")
            x_s = x_s.unsqueeze(dim=0).unsqueeze(dim=0).float()
            pred = torch.sigmoid(UNet(x_s)[0]).cpu().detach().numpy()[0, 0]
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            prediction[i] = pred

    prediction = resize(prediction, (240, 240, 155))

    prediction[prediction <= 0.5] = 0
    prediction[prediction > 0.5] = 1

    mask = prediction.astype(np.uint8)

    brain = x_or*mask

    return brain, mask
        