"""**UNET IMPLEMENTATION**"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)



class DoubleConv(nn.Module):
    """Double Convolution --->(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_value=0.2, GBN=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Dropout(dropout_value),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_value),
            nn.ReLU(inplace=True))
        
        if GBN:
          self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Dropout(dropout_value),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            GhostBatchNorm(out_channels, 1),
            nn.Dropout(dropout_value),
            nn.ReLU(inplace=True))



    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    """Downsampling Channel Size with maxpool followed by Double Conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    """Upsampling followed by Double Convolution"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # First Layer
        self.inc = DoubleConv(n_channels, 16)

        

         # -------------- Mask -------------------
        # self.double_conv = DoubleConv(32, 32, GBN=True) 
        self.down1_mask = DownSample(16, 32)
        self.down2_mask = DownSample(32, 64)
        self.down3_mask = DownSample(64, 128)
        factor = 2 if bilinear else 1
        self.down4_mask = DownSample(128, 256 // factor)
        self.up1_mask = UpSample(256, 128 // factor, bilinear)
        self.up2_mask = UpSample(128, 64 // factor, bilinear)
        self.up3_mask = UpSample(64,32 // factor, bilinear)
        self.up4_mask = UpSample(32, 16, bilinear)
        self.outc_mask = OutConv(16, n_classes)


        # DEPTH

        # self.inc = DoubleConv(n_channels, 32)
        self.down1_depth = DownSample(16, 32)
        self.down2_depth = DownSample(32, 64)
        self.down3_depth = DownSample(64, 128)
        factor = 2 if bilinear else 1
        self.down4_depth = DownSample(128, 256  // factor)
        self.up1_depth = UpSample(256, 128 // factor, bilinear)
        self.up2_depth = UpSample(128, 64 // factor, bilinear)
        self.up3_depth = UpSample(64, 32 // factor, bilinear)
        self.up4_depth = UpSample(32, 16, bilinear)
        self.outc_depth = OutConv(16, n_classes)



    def forward(self, image): # add background and crop & concatenate the same at right position
        _input = image
        x1 = self.inc(_input)

        # MASK

        ## ENCODER
        x2_mask = self.down1_mask(x1)
        x3_mask = self.down2_mask(x2_mask)
        x4_mask = self.down3_mask(x3_mask)
        x5_mask = self.down4_mask(x4_mask)
        
        ## DECODER
        x_mask = self.up1_mask(x5_mask, x4_mask)
        x_mask = self.up2_mask(x_mask, x3_mask)
        x_mask = self.up3_mask(x_mask, x2_mask)

        _mask = self.outc_mask(x_mask)


        # DEPTH 

        ## ENCODER
        x2_depth = self.down1_depth(x1)
        x3_depth = self.down2_depth(x2_depth)
        x4_depth = self.down3_depth(x3_depth)
        x5_depth = self.down4_depth(x4_depth)
        
        ## DECODER
        x_depth = self.up1_depth(x5_depth, x4_depth)
        x_depth = self.up2_depth(x_depth, x3_depth)
        x_depth = self.up3_depth(x_depth, x2_depth)
    
        _depth = self.outc_depth(x_depth)
        # print('_depth map shape', _mask.size())

        return _mask, _depth


