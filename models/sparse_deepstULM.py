import torch
import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from torch import nn
import numpy as np

cat = ME.cat
add = ME.MinkowskiUnion()


class UNet(ME.MinkowskiNetwork):

    def __init__(self, in_nchannel, out_nchannel, D):
        super(UNet, self).__init__(D)
        self.block1 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_nchannel,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(16))

        self.block2 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(32),
        )

        self.block3 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(64))

        self.block3_tr = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(16))

        self.block2_tr = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=64,
                out_channels=16,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(16))

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=24,
            out_channels=out_nchannel,
            kernel_size=1,
            stride=1,
            dimension=D)

    def forward(self, x):
        out_s1 = self.block1(x)
        out = MF.relu(out_s1)

        out_s2 = self.block2(out)
        out = MF.relu(out_s2)

        out_s4 = self.block3(out)
        out = MF.relu(out_s4)

        out = MF.relu(self.block3_tr(out))
        out = ME.cat(out, out_s2)

        out = MF.relu(self.block2_tr(out))
        out = ME.cat(out, out_s1)

        return self.conv1_tr(out)


class InputTransition(nn.Module):
    def __init__(self, outChans, dim=3):
        super(InputTransition, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(in_channels=2,
                                             out_channels=outChans,
                                             kernel_size=5,
                                             stride=1,
                                             dimension=dim)
        self.bn1 = ME.MinkowskiBatchNorm(outChans)
        self.relu = ME.MinkowskiELU()

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = ME.cat((x, x, x, x,
                      x, x, x, x))
        out = self.relu(out + x16)
        return out


class DownTransitionMax(nn.Module):
    def __init__(self, inChans, nConvs, dim=3):
        super(DownTransitionMax, self).__init__()
        outChans = 2*inChans
        self.down_conv = ME.MinkowskiConvolution(
            in_channels=inChans,
            out_channels=outChans,
            kernel_size=1,
            stride=1,
            dimension=dim)
        kernel_size = (dim-1) * [2] + [4]
        self.mp1 = ME.MinkowskiMaxPooling(kernel_size=kernel_size,
                                          stride=kernel_size,
                                          dimension=dim)
        self.bn1 = ME.MinkowskiBatchNorm(outChans)
        self.relu = ME.MinkowskiELU()
        self.ops = _make_nConv(outChans, nConvs, kern=3, dim=dim)

    def forward(self, x):
        down = self.mp1(self.relu(self.bn1(self.down_conv(x))))
        out = self.ops(down)
        out = self.relu(out + down)
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans,
                 outChans, nConvs,
                 temp_mp_skip, dim=3):
        super(UpTransition, self).__init__()
        kernel_size = (dim-1) * [2] + [1]
        stride = (dim-1) * [2] + [1]
        self.up = ME.MinkowskiConvolutionTranspose(in_channels=inChans,
                                                   out_channels=inChans,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   dimension=dim,
                                                   bias=False)

        self.up_conv = ME.MinkowskiConvolution(
            in_channels=inChans,
            out_channels=outChans // 2,
            kernel_size=3,
            stride=1,
            dimension=dim)
        kernel_size = (dim-1) * [1] + [temp_mp_skip]
        self.mp1 = ME.MinkowskiMaxPooling(kernel_size=kernel_size,
                                          stride=kernel_size,
                                          dimension=dim)
        self.bn1 = ME.MinkowskiBatchNorm(outChans // 2)
        self.relu = ME.MinkowskiELU()
        self.ops = _make_nConv(outChans,  nConvs, kern=3, dim=dim)

    def forward(self, x, skipx):
        skipx = self.mp1(skipx)
        out = self.relu(self.bn1(self.up_conv(self.up(x))))
        xcat = ME.cat((out, skipx))
        out = self.ops(xcat)
        out = self.relu(out + xcat)
        return out


class Transition1(ME.MinkowskiNetwork):
    def __init__(self, inChans, outChans, nConvs,
                 mpool_kernel_size=(1, 1, 4), mpool_kernel_stride=(1, 1, 4),
                 dim=3):
        super(Transition1, self).__init__(D=dim)
        self.down_conv = ME.MinkowskiConvolution(in_channels=inChans,
                                                 out_channels=outChans,
                                                 kernel_size=1,
                                                 stride=1,
                                                 dimension=dim,
                                                 bias=False)
        self.mp1 = ME.MinkowskiMaxPooling(kernel_size=mpool_kernel_size,
                                          stride=mpool_kernel_stride,
                                          dimension=dim)
        self.bn1 = ME.MinkowskiBatchNorm(outChans)
        self.relu1 = ME.MinkowskiELU()
        self.relu2 = ME.MinkowskiELU()
        self.ops = _make_nConv(outChans, nConvs, kern=3, dim=dim)

    def forward(self, x):
        down = self.mp1(self.relu1(self.bn1(self.down_conv(x))))
        out = self.ops(down)
        out = self.relu2(add(out, down))
        return out


class Transition2(ME.MinkowskiNetwork):
    def __init__(self, inChans, nConvs,
                 mpool_kernel_size=(1, 1, 4),
                 mpool_kernel_stride=(1, 1, 4),
                 dim=3):
        super(Transition2, self).__init__(D=dim)
        self.down_conv = ME.MinkowskiConvolution(in_channels=inChans,
                                                 out_channels=inChans,
                                                 kernel_size=1,
                                                 stride=1,
                                                 dimension=dim,
                                                 bias=False)
        self.bn1 = ME.MinkowskiBatchNorm(inChans)
        self.mp1 = ME.MinkowskiMaxPooling(kernel_size=mpool_kernel_size,
                                          stride=mpool_kernel_stride,
                                          dimension=dim)

        self.relu1 = ME.MinkowskiELU()
        self.relu2 = ME.MinkowskiELU()
        self.ops = _make_nConv(inChans, nConvs, kern=3, dim=dim)

    def forward(self, x):
        down = self.mp1(self.relu1(self.bn1(self.down_conv(x))))
        out = self.ops(down)
        out = self.relu2(add(out, down))
        return out


class OutputTransition(ME.MinkowskiNetwork):
    def __init__(self, inChans, nConvs, dim=3):
        super(OutputTransition, self).__init__(D=dim)
        self.n_conv = nConvs

        if self.n_conv > 0:
            outChans = inChans // 2
            self.conv1 = ME.MinkowskiConvolution(in_channels=inChans,
                                                 out_channels=outChans,
                                                 kernel_size=3,
                                                 stride=1,
                                                 dimension=dim,
                                                 bias=False)
            self.bn1 = ME.MinkowskiBatchNorm(outChans)
            self.relu1 = ME.MinkowskiELU()

            self.ops = _make_nConv(outChans, nConvs, kern=3, dim=dim)
            self.relu2 = ME.MinkowskiELU()
        else:
            outChans = inChans
        self.conv_out = ME.MinkowskiConvolution(
            outChans, 1, kernel_size=1, dimension=dim)

    def forward(self, x):
        if self.n_conv > 0:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.ops(x))

        out = self.conv_out(x)

        return out


class LUConv(ME.MinkowskiNetwork):
    def __init__(self, nchan, kern, dim=3):
        super(LUConv, self).__init__(D=dim)
        self.relu1 = ME.MinkowskiELU()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=kern,
            stride=1,
            dimension=dim,
            bias=False)
        self.bn1 = ME.MinkowskiBatchNorm(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, kern=3, dim=3):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, kern, dim=dim))
    return nn.Sequential(*layers)


class HRTransition(ME.MinkowskiNetwork):
    def __init__(self, inChans, nConvs, dim=3, pruning=False,
                 deep_supervision=False, kernel_size=3):
        super(HRTransition, self).__init__(D=dim)
        if pruning:
            self.deep_supervision = True
        else:
            self.deep_supervision = deep_supervision
        self.pruning = pruning
        if deep_supervision:
            self.activation_pruning = ME.MinkowskiSigmoid()
        if pruning:
            self.pruning_layer = ME.MinkowskiPruning()

            try:
                self.pruning_threshold = pruning['threshold']
            except TypeError:
                self.pruning_threshold = 0.5

        up_conv = ME.MinkowskiConvolutionTranspose(in_channels=inChans,
                                                   out_channels=inChans,
                                                   kernel_size=(
                                                       dim-1) * [2] + [1],
                                                   stride=(dim-1) * [2] + [1],
                                                   dimension=dim,
                                                   bias=False)

        bn1 = ME.MinkowskiBatchNorm(inChans)
        relu1 = ME.MinkowskiELU()
        self.up_block = nn.Sequential(up_conv, bn1, relu1)

        self.ops = _make_nConv(inChans, nConvs,  kern=3, dim=dim)
        if self.deep_supervision:
            self.block_cls = ME.MinkowskiConvolution(
                inChans, 1, kernel_size=kernel_size, bias=True, dimension=dim)
        self.relu2 = ME.MinkowskiELU()

    def forward(self, x):
        if self.deep_supervision:
            out_pruning = self.block_cls(x)
            out_pruning = self.activation_pruning(out_pruning)
        if self.pruning:
            # check to avoid empty tensor generation
            # if max is superior to pruning threshold proceed to pruning
            # else only keep max values
            if torch.max(out_pruning.F) > self.pruning_threshold:
                keep1 = (out_pruning.F > self.pruning_threshold).squeeze()
            else:
                keep1 = (out_pruning.F >= torch.max(
                    out_pruning.F.detach())).squeeze()
            x = self.pruning_layer(x, keep1)
        up = self.up_block(x)

        out = self.ops(up)
        out = self.relu2(add(out, up))
        if self.deep_supervision:
            return out, out_pruning
        else:
            return out


class SparseULMunet(ME.MinkowskiNetwork):
    def __init__(self, n_base_channels=8, unet=True,
                 n_chan_in=2,
                 transition_1={"inChans": 32,
                               "outChans": 16,
                               "nConvs": 2},
                 hr_transition={"inChans": 16,
                                "nConvs": 3},
                 transition_2={"inChans": 16,
                               "nConvs": 2},
                 output_transition={"inChans": 16,
                                    "nConvs": 2},
                 temporal_mp_kern_sizes=[8, 8, 8],
                 spatial_mp_kern_size=[1, 1, 1],
                 temporal_mp_kern_strides=None,
                 pruning=False,
                 multiphase=False,
                 deep_supervision=False,
                 dim=3, stride=8):
        super(SparseULMunet, self).__init__(dim)
        print(dim)
        if temporal_mp_kern_strides is None:
            temporal_mp_kern_strides = temporal_mp_kern_sizes
        self.spatial_mp_kern_size = spatial_mp_kern_size
        self.dim = dim
        if pruning:
            self.deep_supervision = True
        else:
            self.deep_supervision = deep_supervision

        self.stride = stride
        self.pruning = pruning

        self.tmp_mp_kern_sizes = temporal_mp_kern_sizes
        self.temporal_mp_kern_strides = temporal_mp_kern_strides
        self.n_frames = 512

        self.in_tr = InputTransition(16, dim=dim)
        self.down_tr64 = DownTransitionMax(16, 3, dim=dim)
        self.down_tr128 = DownTransitionMax(32, 3, dim=dim)
        self.up_tr128 = UpTransition(64, 64, 3, temp_mp_skip=4, dim=dim)
        self.up_tr64 = UpTransition(64, 32, 3, temp_mp_skip=16, dim=dim)

        # We allow unstrided maxpool in spatial direction
        tmp_mp_0_kern_size = (dim-1) * [self.spatial_mp_kern_size[0]]
        tmp_mp_0_kern_size += [self.tmp_mp_kern_sizes[0]]
        tmp_mp_0_kern_stride = (
            dim-1) * [1] + [self.temporal_mp_kern_strides[0]]
        self.tmp_mp_0 = ME.MinkowskiMaxPooling(kernel_size=tmp_mp_0_kern_size,
                                               stride=tmp_mp_0_kern_stride,
                                               dimension=dim)

        tr1_mp_kern_size = (dim-1) * [self.spatial_mp_kern_size[1]]
        tr1_mp_kern_size += [self.tmp_mp_kern_sizes[1]]
        tr1_mp_kern_stride = (dim-1) * [1] + [self.temporal_mp_kern_strides[1]]
        self.tr1 = Transition1(**transition_1,
                               mpool_kernel_size=tr1_mp_kern_size,
                               mpool_kernel_stride=tr1_mp_kern_stride,
                               dim=dim)

        self.hr_tr1 = HRTransition(
            **hr_transition, dim=dim, pruning=pruning,
            deep_supervision=self.deep_supervision)
        self.hr_tr2 = HRTransition(
            **hr_transition, dim=dim, pruning=pruning,
            deep_supervision=self.deep_supervision)
        self.hr_tr3 = HRTransition(
            **hr_transition, dim=dim, pruning=pruning,
            deep_supervision=self.deep_supervision)

        tr2_mp_kern_size = (dim-1) * [self.spatial_mp_kern_size[2]] 
        tr2_mp_kern_size += [self.tmp_mp_kern_sizes[2]]
        tr2_mp_kern_stride = (dim-1) * [1] + [self.temporal_mp_kern_strides[2]]
        self.tr2 = Transition2(**transition_2,
                               mpool_kernel_size=tr2_mp_kern_size,
                               mpool_kernel_stride=tr2_mp_kern_stride,
                               dim=dim)
        self.out_tr = OutputTransition(**output_transition, dim=dim)
        self.activation = ME.MinkowskiSigmoid()
        if multiphase:
            self.current_phase = 0
        else:
            self.current_phase = 3

        # Temporal dimension is decreased before the upsampling and after
        # therefore the intermediate outputs have the same temporal stride
        # This is architecture dependent

        self.temporal_strides = 3 * \
            [np.prod(temporal_mp_kern_strides[:2])] + \
            [np.prod(temporal_mp_kern_strides)]
        self.spatial_strides = [8, 4, 2, 1]

    def forward(self, x):
        out32 = self.in_tr(x)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.tr1(out)
        if self.deep_supervision or self.pruning:
            outputs_pruning = []
            out, output_pruning = self.hr_tr1(out)
            outputs_pruning.append(output_pruning)
            if self.current_phase > 0:
                out, output_pruning = self.hr_tr2(out)
                outputs_pruning.append(output_pruning)
            if self.current_phase > 1:
                out, output_pruning = self.hr_tr3(out)
                outputs_pruning.append(output_pruning)
        else:
            out = self.hr_tr1(out)
            out = self.hr_tr2(out)
            out = self.hr_tr3(out)
        if self.current_phase > 2:
            out = self.tr2(out)
            out = self.out_tr(out)
            out = self.activation(out)
        if self.deep_supervision:
            return out, outputs_pruning
        else:
            return out

    def deactivate_deep_supervision(self):
        if self.pruning:
            self.deep_supervision = False
        else:
            for module in self.modules():
                if hasattr(module, 'deep_supervision'):
                    module.deep_supervision = False
