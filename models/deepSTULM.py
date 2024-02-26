"""Credits to Leo Milecki"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=False)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, kern, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=kern, padding=kern//2)
        self.bn1 = nn.BatchNorm3d(nchan)  # ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu, kern=4):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, kern, elu))
    return nn.Sequential(*layers)


class LUConv2D(nn.Module):
    def __init__(self, nchan, kern, elu):
        super(LUConv2D, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=kern, padding=kern//2)
        self.bn1 = nn.BatchNorm2d(nchan)  # ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv2D(nchan, depth, elu, kern=4):
    layers = []
    for _ in range(depth):
        layers.append(LUConv2D(nchan, kern, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(2, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(outChans)  # ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        out = self.bn1(self.conv1(x))

        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x,
                         x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransitionMax(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransitionMax, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=1, padding=0)
        self.mp1 = nn.MaxPool3d(kernel_size=(4, 2, 2))
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu, kern=3)

    def forward(self, x):
        down = self.mp1(self.relu1(self.bn1(self.down_conv(x))))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.up_conv = nn.Conv3d(
            inChans, outChans // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu, kern=3)

    def forward(self, x, skipx):
        out = self.do1(x)
        mp = skipx.size()[2]//32
        skipx = F.max_pool3d(skipx, kernel_size=(mp, 1, 1))
        out = self.relu1(self.bn1(self.up_conv(self.up(out))))
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class Transition1(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(Transition1, self).__init__()
        outChans = inChans // 2
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=1, padding=0)
        self.mp1 = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu, kern=3)

    def forward(self, x):
        down = self.mp1(self.relu1(self.bn1(self.down_conv(x))))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class HRTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(HRTransition, self).__init__()
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.up_conv = nn.Conv3d(inChans, inChans, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(inChans)
        self.do1 = passthrough
        self.relu1 = nn.ELU()
        self.relu2 = nn.ELU()
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(inChans, nConvs, elu, kern=3)

    def forward(self, x):
        up = self.relu1(self.bn1(self.up_conv(self.up(x))))
        out = self.do1(up)
        out = self.ops(out)
        out = self.relu2(torch.add(out, up))
        return out


class Transition2(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(Transition2, self).__init__()
        self.down_conv = nn.Conv3d(inChans, inChans, kernel_size=1, padding=0)
        self.mp1 = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.bn1 = nn.BatchNorm3d(inChans)  # ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, inChans)
        self.relu2 = ELUCons(elu, inChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(inChans, nConvs, elu, kern=3)

    def forward(self, x):
        down = self.mp1(self.relu1(self.bn1(self.down_conv(x))))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu):
        super(OutputTransition, self).__init__()
        outChans = inChans // 2

        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(outChans)  # ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.mp1 = nn.MaxPool3d((2, 1, 1))

        self.ops = _make_nConv2D(outChans, nConvs, elu, kern=3)
        self.relu2 = ELUCons(elu, outChans)

        self.conv_out = nn.Conv2d(outChans, 1, kernel_size=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.mp1(out)

        out = torch.squeeze(out, dim=2)
        out = self.relu2(self.ops(out))

        out = self.conv_out(out)

        return out

# ulmVnet with maxpool layers in encoder


class ulmVNetMax(nn.Module):
    def __init__(self, elu=True, dropout=False):
        super(ulmVNetMax, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr64 = DownTransitionMax(16, 3, elu, dropout=dropout)
        self.down_tr128 = DownTransitionMax(32, 3, elu, dropout=dropout)
        self.up_tr128 = UpTransition(64, 64, 3, elu, dropout=dropout)
        self.up_tr64 = UpTransition(64, 32, 3, elu, dropout=dropout)
        self.tr1 = Transition1(32, 2, elu, dropout=dropout)
        self.hr_tr1 = HRTransition(16, 3, elu, dropout=dropout)
        self.hr_tr2 = HRTransition(16, 3, elu, dropout=dropout)
        self.hr_tr3 = HRTransition(16, 3, elu, dropout=dropout)
        self.tr2 = Transition2(16, 2, elu, dropout=dropout)
        self.out_tr = OutputTransition(16, 2, elu)

    def forward(self, x):
        out32 = self.in_tr(x)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.tr1(out)
        out = self.hr_tr1(out)
        out = self.hr_tr2(out)
        out = self.hr_tr3(out)
        out = self.tr2(out)
        out = self.out_tr(out)
        return out


class DenseULM(nn.Module):
    def __init__(self, elu=True, dropout=False):
        super(DenseULM, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr64 = DownTransitionMax(16, 3, elu, dropout=dropout)
        self.down_tr128 = DownTransitionMax(32, 3, elu, dropout=dropout)
        self.up_tr128 = UpTransition(64, 64, 3, elu, dropout=dropout)
        self.up_tr64 = UpTransition(64, 32, 3, elu, dropout=dropout)
        self.tr1 = Transition1(32, 2, elu, dropout=dropout)

    def forward(self, x):
        out32 = self.in_tr(x)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.tr1(out)
        return out
