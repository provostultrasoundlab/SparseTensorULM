import torch
import torch.nn as nn


class SeparableConvST(nn.Module):
    """
    Separable convolution for spatiotemporal data described on [1]. 
    Spatial convolution is performed first, temporal one in second

    [1]D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, 
    “A Closer Look at Spatiotemporal Convolutions for Action Recognition,” 
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 
    Jun. 2018, pp. 6450–6459. doi: 10.1109/CVPR.2018.00675.

    """

    def __init__(self, n_chans, kernel_size, stride, dim,  groups, bias):
        super(SeparableConvST, self).__init__()
        if dim == 3:
            self.spatial_conv = nn.Conv2d(n_chans, n_chans,
                                          kernel_size=kernel_size,
                                          padding=kernel_size//2,
                                          groups=groups,
                                          stride=stride, bias=bias)
        elif dim == 4:
            self.spatial_conv = nn.Conv3d(n_chans, n_chans,
                                          kernel_size=kernel_size,
                                          padding=kernel_size//2,
                                          groups=groups,
                                          stride=stride, bias=bias)
        self.temporal_conv = nn.Conv1d(n_chans, n_chans,
                                       kernel_size=kernel_size,
                                       padding=kernel_size//2, groups=groups,
                                       stride=stride, bias=bias)

    def forward(self, x):
        # x shape is BxCxHxW(xD)xT
        # dims index = 1,2,3,4,5 or 1,2,3,4,5,6
        x_shape = x.shape
        dims = tuple(range(len(x_shape)))
        # we apply the spatial convolution
        # dims index = 1,2,5,4,5 or 1,2,3,4,5,6
        new_dims = dims[:1]+dims[-1:] + dims[1:-1]
        x = torch.permute(x, new_dims)
        # merge temporal and batch dimension
        new_shape = tuple((-1, *x_shape[1:2],
                           *x_shape[2:-1]))
        x = x.reshape(new_shape)
        x = self.spatial_conv(x)
        new_shape = (
            x_shape[0:1] + x_shape[-1:] + x_shape[1:-1])
        x = x.reshape(*new_shape)
        new_dims = dims[:1] + dims[2:3] + dims[3:] + dims[1:2]
        x = torch.permute(x, new_dims)
        return x


convST = SeparableConvST(4, 3, 1, 3, 2, True)
inp = torch.randint(0, 10, (2, 4, 32, 64, 10)).float()
out = convST(inp)


class ConvNeXtBlock(nn.Module):
    # Conv Block based on
    # [1]Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie,
    # “A ConvNet for the 2020s,” 2022, pp. 11976–11986.
    def __init__(self, n_chan_in=96, n_chan_out=96,
                 n_chan_middle=384, kernel_size=7,
                 activation=nn.GELU(),
                 normalization=nn.LayerNorm, maxpool=[1, 1],
                 dim=3):
        super(ConvNeXtBlock, self).__init__()
        self.normalization = normalization(n_chan_in)
        self.activation = activation()
        self.in_conv = SeparableConvST(n_chans=n_chan_in,
                                       kernel_size=kernel_size,
                                       stride=1, bias=True, groups=n_chan_in,
                                       dim=dim)

        self.mid_conv = nn.Conv1d(in_channels=n_chan_in,
                                  out_channels=n_chan_middle,
                                  kernel_size=1,
                                  stride=1,
                                  bias=True)
        self.out_conv = nn.Conv1d(in_channels=n_chan_middle,
                                  out_channels=n_chan_out,
                                  kernel_size=1,
                                  stride=1,
                                  bias=True)
        self.mp_stride = maxpool
        assert maxpool[0] == 1
        kernel_size = tuple([maxpool[0]] + [maxpool[1]])
        padding = tuple([maxpool[0]//2] + [(maxpool[1]-1)//2])
        self.maxpool = nn.MaxPool2d(kernel_size,
                                    padding=padding)
        self.n_chan_in = n_chan_in
        self.n_chan_middle = n_chan_middle
        self.n_chan_out = n_chan_out

    def forward(self, x):
        input = x
        x = self.in_conv(x)
        x_shape = x.shape
        x = self.normalization(x.reshape(*x_shape[:2], -1))
        x = x.reshape(x_shape)
        x_shape = x.shape
        x = self.mid_conv(x.reshape(*x_shape[:2], -1))
        x = x.reshape(x_shape[0], self.n_chan_middle, *x_shape[2:])
        x = self.activation(x)
        x_shape = x.shape
        x = self.out_conv(x.reshape(*x_shape[:2], -1))
        x = x.reshape(x_shape[0], self.n_chan_out, *x_shape[2:])
        if self.n_chan_in == self.n_chan_out:
            x = x + input
        x_shape = x.shape
        x = x.reshape(*x_shape[:2], -1, x_shape[-1])
        x = self.maxpool(x)
        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x


class DenseToSparse(nn.Module):
    def __init__(self, n_chan_in=2, conv_blocks=None, dim=3,
                 normalization='layer',
                 int_activation_fn='relu',
                 decision_activation_fn='sigmoid'):
        super(DenseToSparse, self).__init__()
        self.dim = dim
        if int_activation_fn == 'relu':
            int_activation_fn = nn.ReLU
        elif int_activation_fn == 'prelu':
            int_activation_fn = nn.PReLU
        elif int_activation_fn == 'gelu':
            int_activation_fn = nn.GELU
        else:
            raise NotImplementedError('only ReLU, PReLU and GeLU activation')

        if decision_activation_fn == 'sigmoid':
            decision_activation_fn = nn.Sigmoid
        else:
            raise NotImplementedError('only sigmoid activation')

        if normalization == 'layer':
            normalization = nn.LayerNorm
        elif normalization == 'instance':
            normalization = nn.InstanceNorm1d
        else:
            raise NotImplementedError('only layer normalization')

        self.final_activation = decision_activation_fn()
        self.conv_blocks = nn.ModuleList()
        for conv_blocks_args in conv_blocks:
            self.conv_blocks.append(ConvNeXtBlock(**conv_blocks_args,
                                                  normalization=normalization,
                                                  activation=int_activation_fn,
                                                  dim=dim))
        self.final_conv = nn.Conv1d(in_channels=conv_blocks[-1]['n_chan_out'],
                                    out_channels=1, kernel_size=1)
        self.stride = 1
        self.temporal_strides = 1
        for block in self.conv_blocks:
            self.temporal_strides *= block.mp_stride[-1]

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x_shape = x.shape
        # flatten spatiotemporal dim of x keeping batch and chan dims
        x = self.final_conv(x.reshape(*x_shape[:2], -1))
        x = self.final_activation(x)
        x = x.reshape(x_shape[0], 1, *x_shape[2:])
        return x
