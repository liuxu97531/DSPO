import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT
from .mlp import MLP
from .rff import PositionalEncoding


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 2
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, size=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        if size is not None:
            out_ft = torch.zeros(batchsize, self.out_channels, size[0], size[1] // 2 + 1, dtype=torch.cfloat,
                                 device=x.device)
        else:
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                                 device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if size is not None:
            x = torch.fft.irfft2(out_ft, s=(size[0], size[1]))
        else:
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNORecon(nn.Module):
    def __init__(self, sensor_num, fc_size, out_size, modes1=24, modes2=24, width=32):
        super(FNORecon, self).__init__()
        self.fc_size = fc_size
        self.out_size = out_size
        self.fc = nn.Sequential(
            nn.Linear(sensor_num, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, fc_size[0] * fc_size[1]),
        )
        self.conv_smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.InstanceNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.InstanceNorm2d(width),
            nn.GELU(),
        )
        self.embedding = nn.Conv2d(1, width, kernel_size=1)
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        N, _ = x.shape
        x = self.fc(x).reshape(N, 1, self.fc_size[0], self.fc_size[1])
        x = self.embedding(x)
        x = self.conv_smooth(F.interpolate(x, scale_factor=2))

        x = F.interpolate(x, self.out_size)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)


class FNO2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, modes1=12, modes2=12, width=32):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        2. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=2)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels + 2, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


""" Def: 2d Wavelet layer """


class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.dwt_ = DWT(J=self.level, mode='symmetric', wave='db4').to(dummy.device)
        self.mode_data, _ = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-2]
        self.modes2 = self.mode_data.shape[-1]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT(J=self.level, mode='symmetric', wave='db4').to(x.device)
        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:, :, 0, :, :] = self.mul2d(x_coeff[-1][:, :, 0, :, :].clone(), self.weights2)
        x_coeff[-1][:, :, 1, :, :] = self.mul2d(x_coeff[-1][:, :, 1, :, :].clone(), self.weights3)
        x_coeff[-1][:, :, 2, :, :] = self.mul2d(x_coeff[-1][:, :, 2, :, :].clone(), self.weights4)

        # Return to physical space
        idwt = IDWT(mode='symmetric', wave='db4').to(x.device)
        x = idwt((out_ft, x_coeff))
        return x


""" The forward operation """


class WNO2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=64, level=4, dummy_data=torch.randn(1, 1, 200, 200),
                 padding=0):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        2. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+2) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=2)
        """

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = padding  # pad the domain when required
        self.fc0 = nn.Linear(in_channels + 2, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 192)
        self.fc2 = nn.Linear(192, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])
        x = F.pad(x, [1, 0, 0, 0])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        if self.padding > 0:
            x = x[..., :, 1:]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class WNORecon(nn.Module):
    def __init__(self, sensor_num, fc_size, out_size, out_channels=1, width=64, level=4,
                 dummy_data=torch.randn(1, 1, 200, 200),
                 padding=0):
        super(WNORecon, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        2. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+2) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=2)
        """
        self.fc_size = fc_size
        self.out_size = out_size
        self.fc = nn.Sequential(
            nn.Linear(sensor_num, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, fc_size[0] * fc_size[1]),
        )
        self.conv_smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.InstanceNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.InstanceNorm2d(width),
            nn.GELU(),
        )
        self.embedding = nn.Conv2d(1, width, kernel_size=1)
        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 0  # pad the domain when required
        self.fc0 = nn.Linear(width + 2, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 192)
        self.fc2 = nn.Linear(192, out_channels)

    def forward(self, x):
        N, _ = x.shape
        x = self.fc(x).reshape(N, 1, self.fc_size[0], self.fc_size[1])
        x = self.embedding(x)
        x = self.conv_smooth(F.interpolate(x, scale_factor=2))

        x = F.interpolate(x, self.out_size)
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        B = B.to(x.device)
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DeepONet(nn.Module):
    def __init__(self, layer_size_branch=[32, 128, 128, 128, 128],
                 layer_size_trunk=[1, 128, 128, 128], fourier_feature=False):
        super(DeepONet, self).__init__()
        self.branch_net = MLP(layers=layer_size_branch)
        self.trunk_net = MLP(layers=layer_size_trunk)
        self.fourier_feature = fourier_feature
        if fourier_feature:
            self.position_encoding = PositionalEncoding(sigma=1.0, m=64)

    def forward(self, x_branch, x_trunk):
        # Branch net to encode the input function
        y_branch = self.branch_net(x_branch)
        # Trunk net to encode the domain of the output function
        if self.fourier_feature:
            x_trunk = self.position_encoding(x_trunk)
        y_trunk = self.trunk_net(x_trunk)
        # Dot product
        if y_branch.shape[-1] != y_trunk.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        Y = torch.einsum("bi,ni->bn", y_branch, y_trunk)
        return Y

if __name__ == '__main__':
    # x_branch, x_trunk = torch.randn(8, 32), torch.randn(40000, 2)
    # net = DeepONet(layer_size_branch=[32, 256, 256, 256, 256],
    #                layer_size_trunk=[256, 256, 256, 256], fourier_feature=True)
    # with torch.no_grad():
    #     y = net(x_branch, x_trunk)
    # print(y.shape)

    # # test deeponet
    x_branch, x_trunk = torch.randn(2000, 32), torch.randn(2000, 2)
    net = DeepONet(layer_size_branch=[32, 256, 256, 256, 256],
                   layer_size_trunk=[2, 256, 256, 256], fourier_feature=False)
    with torch.no_grad():
        y = net(x_branch, x_trunk)
    print(y.shape)