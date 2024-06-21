"""Model and module definitions
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from collections import OrderedDict


"""Inspire by the pytorch implementation of ResNet
https://github.com/pytorch/vision/blob/aa21197462591a89f527ded3938ffe62f7d8cb8f/torchvision/models/resnet.py
"""
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, activation_fn = None, batch_normalization=False) -> nn.Conv2d:
    """3x3 convolution with padding and with optional batch normalization and activation"""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        ),
        nn.BatchNorm2d(out_planes) if batch_normalization else ...,
        activation_fn() if activation_fn is not None else ...
    )

def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, activation_fn = None, batch_normalization=False) -> nn.Conv2d:
    """5x5 convolution with padding and with optional batch normalization and activation"""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=5,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        ),
        nn.BatchNorm2d(out_planes) if batch_normalization else ...,
        activation_fn() if activation_fn is not None else ...
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, activation_fn = None, batch_normalization=False) -> nn.Conv2d:
    """1x1 convolution with optional batch normalization and activation"""
    if batch_normalization:
        ...
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes) if batch_normalization else ...,
        activation_fn() if activation_fn is not None else ...
    )

def convNxN(in_planes: int, out_planes: int, kernel_size: int, activation_fn = None, batch_normalization=False) -> nn.Conv2d:
    """NxN convolution with padding and with optional batch normalization and activation"""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=int((kernel_size-1) / 2),
            groups=1,
            bias=False,
            dilation=1,
        ),
        nn.BatchNorm2d(out_planes) if batch_normalization else ...,
        activation_fn() if activation_fn is not None else ...
    )

class InceptionBlock(nn.Module):
    """Inspired by https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html#Inception
    Implements the variant of inception block described in https://papers.nips.cc/paper/6489-single-image-depth-perception-in-the-wild
    Inputs:
            c_in - Number of input feature maps from the previous layers
            kernel_sizes - Dictionary specifying the kernel sizes for convolutions, with keys "Conv1".."Conv4"
            c_red - Dictionary specifying the output of the dimensionality reducing 1x1 convolutions, with keys "Conv2".."Conv4"
            c_out - Dictionary specifying the output number of feature maps for individual branches, with keys "Conv1".."Conv4"
            act_fn - Activation class constructor (e.g. nn.ReLU)
    """
    def __init__(self, c_in : int, kernel_sizes:dict, c_red: dict, c_out : dict, act_fn) -> None:
        super().__init__()

        self.conv1 = conv1x1(c_in, c_out["Conv1"], activation_fn=act_fn, batch_normalization=True)
        self.conv2 = nn.Sequential(
            conv1x1(c_in, c_red["Conv2"], activation_fn=act_fn, batch_normalization=True),
            convNxN(c_red["Conv2"], c_out["Conv2"], kernel_sizes["Conv2"], activation_fn=act_fn, batch_normalization=True)
        )
        self.conv3 = nn.Sequential(
            conv1x1(c_in, c_red["Conv3"], activation_fn=act_fn, batch_normalization=True),
            convNxN(c_red["Conv3"], c_out["Conv3"], kernel_sizes["Conv3"], activation_fn=act_fn, batch_normalization=True)
        )
        self.conv4 = nn.Sequential(
            conv1x1(c_in, c_red["Conv4"], activation_fn=act_fn, batch_normalization=True),
            convNxN(c_red["Conv4"], c_out["Conv4"], kernel_sizes["Conv4"], activation_fn=act_fn, batch_normalization=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x_out = torch.cat([x1, x2, x3, x4], dim=1)
        return x_out

class KNN_Hourglass_InTheWild_Full(nn.Module):
    def __init__(self, in_width, in_height, in_channels) -> None:
        super().__init__()

        self.activation_fn = nn.ReLU

        self.H = nn.Conv2d(in_channels, 128, 3)

        inception_branches_ratios = [1/4, 1/2, 1/8, 1/8]
        kernel_keys = ["Conv2", "Conv3", "Conv4"]
        inter_keys = ["Conv2", "Conv3", "Conv4"]
        out_keys = ["Conv1", "Conv2", "Conv3", "Conv4"]

        A_in = 128
        A_out = 64
        A_kernels = [3, 7, 11]
        A_interDim = 64

        B_in = 128
        B_out = 128
        B_kernels = [3,5,7]
        B_interDim = 32

        C_in = 128
        C_out = 128
        C_interDim = 64
        C_kernels = [3,7,11]

        D_in = 128
        D_out = 256
        D_interDim = 32
        D_kernels = [3,5,7]

        E_in = 256
        E_out = 256
        E_interDim = 32
        E_kernels = [3,5,7]

        F_in = 256
        F_out = 256
        F_interDim = 64
        F_kernels = [3,7,11]

        G_in = 256
        G_out = 128
        G_interDim = 32
        G_kernels = [3,5,7]

        self.A = InceptionBlock(A_in, dict(zip(kernel_keys, A_kernels)), dict(zip(inter_keys, 3*[A_interDim])), dict(zip(out_keys, [int(x) for x in np.multiply(4*[A_out], inception_branches_ratios)])), self.activation_fn)
        self.B = InceptionBlock(B_in, dict(zip(kernel_keys, B_kernels)), dict(zip(inter_keys, 3*[B_interDim])), dict(zip(out_keys, [int(x) for x in np.multiply(4*[B_out], inception_branches_ratios)])), self.activation_fn)
        self.C = InceptionBlock(C_in, dict(zip(kernel_keys, C_kernels)), dict(zip(inter_keys, 3*[C_interDim])), dict(zip(out_keys, [int(x) for x in np.multiply(4*[C_out], inception_branches_ratios)])), self.activation_fn)
        self.D = InceptionBlock(D_in, dict(zip(kernel_keys, D_kernels)), dict(zip(inter_keys, 3*[D_interDim])), dict(zip(out_keys, [int(x) for x in np.multiply(4*[D_out], inception_branches_ratios)])), self.activation_fn)
        self.E = InceptionBlock(E_in, dict(zip(kernel_keys, E_kernels)), dict(zip(inter_keys, 3*[E_interDim])), dict(zip(out_keys, [int(x) for x in np.multiply(4*[E_out], inception_branches_ratios)])), self.activation_fn)
        self.F = InceptionBlock(F_in, dict(zip(kernel_keys, F_kernels)), dict(zip(inter_keys, 3*[F_interDim])), dict(zip(out_keys, [int(x) for x in np.multiply(4*[F_out], inception_branches_ratios)])), self.activation_fn)
        self.G = InceptionBlock(G_in, dict(zip(kernel_keys, G_kernels)), dict(zip(inter_keys, 3*[G_interDim])), dict(zip(out_keys, [int(x) for x in np.multiply(4*[G_out], inception_branches_ratios)])), self.activation_fn)

        # self.B = InceptionBlock(128, {"Conv2":3, "Conv3":7, "Conv4":11}, {"Conv2":32, "Conv3":32, "Conv4":32}, {"Conv1":16, "Conv2":32, "Conv3":8, "Conv4":8})
        # self.C = InceptionBlock(128, {"Conv2":3, "Conv3":7, "Conv4":11}, {"Conv2":64, "Conv3":64, "Conv4":64}, {"Conv1":16, "Conv2":32, "Conv3":8, "Conv4":8})
        # self.D = InceptionBlock(128, {"Conv2":3, "Conv3":7, "Conv4":11}, {"Conv2":64, "Conv3":64, "Conv4":64}, {"Conv1":16, "Conv2":32, "Conv3":8, "Conv4":8})
        # self.E = InceptionBlock(128, {"Conv2":3, "Conv3":7, "Conv4":11}, {"Conv2":64, "Conv3":64, "Conv4":64}, {"Conv1":16, "Conv2":32, "Conv3":8, "Conv4":8})
        # self.F = InceptionBlock(128, {"Conv2":3, "Conv3":7, "Conv4":11}, {"Conv2":64, "Conv3":64, "Conv4":64}, {"Conv1":16, "Conv2":32, "Conv3":8, "Conv4":8})
        # self.G = InceptionBlock(128, {"Conv2":3, "Conv3":7, "Conv4":11}, {"Conv2":64, "Conv3":64, "Conv4":64}, {"Conv1":16, "Conv2":32, "Conv3":8, "Conv4":8})

        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        main = self.H(x)
        main = self.activation_fn(main)
        
        #Downwards

        skip_1 = self.A(main)

        main = self.pool(main)

        main = self.B(main)
        main = self.B(main)

        skip_2 = self.B(main)
        skip_2 = self.C(skip_2)

        main = self.pool(main)

        main = self.B(main)
        main = self.D(main)

        skip_3 = self.E(main)
        skip_3 = self.F(skip_3)

        main = self.pool(main)

        main = self.E(main)
        main = self.E(main)

        skip_4 = self.E(main)
        skip_4 = self.E(skip_4)

        main = self.E(main)
        main = self.E(main)
        main = self.E(main)

        #Upwards

        main = self.upsample(main + skip_4)
        main = self.E(main)
        main = self.F(main)

        main = self.upsample(main + skip_3)
        main = self.E(main)
        main = self.G(main)

        main = self.upsample(main + skip_2)
        main = self.B(main)
        main = self.A(main)

        main = self.upsample(main + skip_1)
        out = self.H(main)

        return out

class KNN_Inception_Hourglass_Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.inception_branches_ratios = [1/4, 1/2, 1/8, 1/8]
        self.kernel_keys = ["Conv2", "Conv3", "Conv4"]
        self.inter_keys = ["Conv2", "Conv3", "Conv4"]
        self.out_keys = ["Conv1", "Conv2", "Conv3", "Conv4"]

        self.act_fn = nn.ReLU

        self.entry = nn.Conv2d(3, 8, 5, padding=2)
        self.exit = nn.Conv2d(8, 1, 5, padding=2)

        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.skip1 = nn.Sequential(
            self.inceptionBlock(8, [3,3,5], 4, 16)
        )

        self.main1 = nn.Sequential(
            self.inceptionBlock(8, [3,3,5], 4, 16),
            # self.inceptionBlock(16, [3,3,5], 4, 16),
        )

        self.skip2 = nn.Sequential(
            self.inceptionBlock(16, [3,3,5], 4, 8),
        )

        self.main2 = nn.Sequential(
            # self.inceptionBlock(16, [3,3,5], 4, 16),
            self.inceptionBlock(16, [3,3,5], 4, 8)
        )

        self.main3 = nn.Sequential(
            self.inceptionBlock(8, [3,3,5], 4, 16),
            # self.inceptionBlock(16, [3,3,5], 4, 16)
        )

        self.main4 = nn.Sequential(
            # self.inceptionBlock(16, [3,3,5], 4, 16),
            self.inceptionBlock(16, [3,3,5], 4, 8)
        )

    def forward(self, x):
        main = self.entry(x)
        skip1 = self.skip1(main)

        main = self.pool(main)
        main = self.main1(main)
        skip2 = self.skip2(main)

        main = self.pool(main)
        main = self.main2(main)

        main = self.upsample(main)
        main = main + skip2
        main = self.main3(main)

        main = self.upsample(main)
        main = main + skip1
        main = self.main4(main)

        out = self.exit(main)
        return out

    def inceptionBlock(self, c_in, kernels, interDim, out) -> nn.Module:
        kernel_sizes = dict(zip(self.kernel_keys, kernels))
        c_red = dict(zip(self.inter_keys, 3*[interDim]))
        c_out = dict(zip(self.out_keys, [int(x) for x in np.multiply(4*[out], self.inception_branches_ratios)]))
        
        return InceptionBlock(c_in, kernel_sizes, c_red, c_out, self.act_fn)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_of_conv_blocks = 1, act_fn = None, pad_to_conserve_dimensions=False) -> None:
        super().__init__()

        padding = 0
        if pad_to_conserve_dimensions:
            padding = int((kernel_size-1) / 2)

        modules = [nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding)]

        for i in range(num_of_conv_blocks-1):
            if act_fn is not None:
                modules.append(act_fn())
            modules.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding = padding))

        self.module = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self.module(x)

"""Inspired by https://amaarora.github.io/2020/09/13/unet.html"""
class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024), kernel_sizes=(3,3,3,3,3,3)):
        super().__init__()
        assert(len(chs) == len(kernel_sizes))

        self.enc_blocks = nn.ModuleList([
            EncoderBlock(chs[i], chs[i+1], kernel_sizes[i], num_of_conv_blocks=2, act_fn=nn.ReLU)
            for i in range(len(chs)-1)
        ])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), kernel_sizes=(3,3,3,3,3,3)):
        super().__init__()
        assert(len(chs) == len(kernel_sizes))

        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([
            EncoderBlock(chs[i], chs[i+1], kernel_sizes[i], num_of_conv_blocks=2, act_fn=nn.ReLU)
            for i in range(len(chs)-1)
        ])
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), kernel_sizes=(3,3,3,3,3,3), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs, kernel_sizes)
        self.decoder     = Decoder(dec_chs, kernel_sizes)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out


# def query_ranking_loss(output, targets):
#     """Loss function evaluating depth map against ordinal depth relations

#     Args:
#         output (_type_): _description_
#         targets (_type_): list of queries (i_k, j_k, r_k) where i_k/j_k are the positions (x, y) of points in the image and r_k \in {-1, 0, 1} is their relation

#     Returns:
#         float: _description_
#     """
#     def single_query_loss(z, i_k, j_k, r_k):
#         if r_k == 1:
#             return math.log2(1+math.exp(-z[i_k] + z[j_k]))
#         elif r_k == -1:
#             return math.log2(1+math.exp(z[i_k] - z[j_k]))
#         elif r_k == 0:
#             return (z[i_k] - z[j_k])**2
#         else:
#             raise ValueError(f"Invalid depth relation {r_k}")

#     r = targets[:,2].reshape(1,-1).t()
#     zeros = torch.zeros_like(r)
#     ones = torch.ones_like(r)

#     t1 = torch.where(r == 1, ones, zeros)
#     t2 = torch.where(r == -1, ones, zeros)
#     t3 = torch.where(r == 0, ones, zeros)

#     loss = 

#     # loss = torch.sum(outputs)

#     return loss

# class QueryRankingLoss(nn.Module):
#   def __init__(self):
#     super(QueryRankingLoss, self).__init__()

#   def forward(self, predictions, target):
#     square_difference = torch.square(predictions - target)
#     loss_value = torch.mean(square_difference)
#     return loss_value