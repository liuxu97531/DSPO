import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.mlp import MLP


class AutoEncoderMLP(nn.Module):
    def __init__(self, encoder_layers=[112 * 192, 4800, 1280, 128, 10],
                 decoder_layers=[10, 128, 1280, 4800, 112 * 192]):
        super(AutoEncoderMLP, self).__init__()
        self.encoder = MLP(layers=encoder_layers)
        self.decoder = MLP(layers=decoder_layers)

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, polling=True):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class AutoEncoderCNN(nn.Module):
    def __init__(self, in_channels=1, encoder_block=[32, 64, 128, 128, 128],
                 decoder_block=[128, 128, 64, 32, 32],
                 encoder_layers=[128 * 3 * 6, 256, 10],
                 decoder_layers=[10, 256, 128 * 3 * 6]):
        super(AutoEncoderCNN, self).__init__()
        self.encoder_cnn1 = _EncoderBlock(in_channels=in_channels, out_channels=encoder_block[0], polling=True)
        self.encoder_cnn2 = _EncoderBlock(in_channels=encoder_block[0], out_channels=encoder_block[1], polling=True)
        self.encoder_cnn3 = _EncoderBlock(in_channels=encoder_block[1], out_channels=encoder_block[2], polling=True)
        self.encoder_cnn4 = _EncoderBlock(in_channels=encoder_block[2], out_channels=encoder_block[3], polling=True)
        self.encoder_cnn5 = _EncoderBlock(in_channels=encoder_block[3], out_channels=encoder_block[4], polling=True)
        self.encoder_cnn6 = nn.Sequential(
            nn.Conv2d(encoder_block[4], encoder_block[4], kernel_size=1),
            nn.GroupNorm(32, encoder_block[4]),
            nn.GELU(),
            nn.Conv2d(encoder_block[4], 1, kernel_size=1)
        )
        self.encoder_fnns = MLP(encoder_layers)
        self.decoder_cnn6 = nn.Sequential(
            nn.Conv2d(1, encoder_block[4], kernel_size=1),
            nn.GroupNorm(32, encoder_block[4]),
            nn.GELU(),
            nn.Conv2d(encoder_block[4], encoder_block[4], kernel_size=1),
            nn.GroupNorm(32, encoder_block[4]),
            nn.GELU(),
        )
        self.decoder_cnn5 = _DecoderBlock(in_channels=encoder_block[4], out_channels=decoder_block[0])
        self.decoder_cnn4 = _DecoderBlock(in_channels=decoder_block[0], out_channels=decoder_block[1])
        self.decoder_cnn3 = _DecoderBlock(in_channels=decoder_block[1], out_channels=decoder_block[2])
        self.decoder_cnn2 = _DecoderBlock(in_channels=decoder_block[2], out_channels=decoder_block[3])
        self.decoder_cnn1 = _DecoderBlock(in_channels=decoder_block[3], out_channels=decoder_block[4])
        self.decoder_fnns = MLP(decoder_layers)
        self.final = nn.Sequential(
            nn.Conv2d(decoder_block[4], decoder_block[4], kernel_size=1),
            nn.GELU(),
            nn.Conv2d(decoder_block[4], 1, kernel_size=1)
        )
        self.layer_size = None

    def forward(self, x):
        h = self.encode(x)
        y = self.decode(h)
        return y

    def encode(self, x):
        y1 = self.encoder_cnn1(x)
        y2 = self.encoder_cnn2(y1)
        y3 = self.encoder_cnn3(y2)
        y4 = self.encoder_cnn4(y3)
        y5 = self.encoder_cnn6(self.encoder_cnn5(y4))
        h = self.encoder_fnns(y5.view(y5.shape[0], -1))
        self.layer_size = [y5.shape, y4.shape, y3.shape, y2.shape, y1.shape]
        return h

    def decode(self, h):
        sizes = self.layer_size
        h = self.decoder_fnns(h)
        z5 = self.decoder_cnn5(self.decoder_cnn6(h.view(h.shape[0], sizes[0][1], sizes[0][2], sizes[0][3])))
        z4 = self.decoder_cnn4(F.interpolate(z5, sizes[1][-2:]))
        z3 = self.decoder_cnn3(F.interpolate(z4, sizes[2][-2:]))
        z2 = self.decoder_cnn2(F.interpolate(z3, sizes[3][-2:]))
        z1 = self.decoder_cnn1(F.interpolate(z2, sizes[4][-2:]))
        y = self.final(z1)
        return y


class AutoEncoderDataset(Dataset):
    def __init__(self, data):
        """
        用于自动编码器的Dataset
        :param index:
        """
        super(AutoEncoderDataset, self).__init__()
        self.data = torch.from_numpy(data).float()

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


def transform(net, data):
    device = next(net.parameters()).device
    data_loader = DataLoader(AutoEncoderDataset(data), batch_size=8, shuffle=False)
    X_t = []
    with torch.no_grad():
        for i, (inputs) in tqdm(enumerate(data_loader)):
            X_t.append(net.encode(inputs.to(device)))
    X_t = torch.cat(X_t, dim=0).cpu().numpy()
    return X_t


def inverse_transform(net, data):
    device = next(net.parameters()).device
    data_loader = DataLoader(AutoEncoderDataset(data), batch_size=8, shuffle=False)
    y = []
    with torch.no_grad():
        for i, (inputs) in tqdm(enumerate(data_loader)):
            y.append(net.decode(inputs.to(device)))
    y = torch.cat(y, dim=0).cpu().numpy()
    return y


if __name__ == '__main__':
    model = AutoEncoderCNN(in_channels=1, encoder_block=[32, 32, 32, 32, 32],
                           decoder_block=[32, 32, 32, 32, 32],
                           encoder_layers=[5 * 11, 32, 10],
                           decoder_layers=[10, 32, 5 * 11]).cuda()
    print(model)
    x = torch.randn(1, 1, 180, 360).cuda()
    with torch.no_grad():
        y = model(x)
    print(y.shape)
    X_t = transform(model, torch.randn(100, 1, 180, 360).numpy())
    print(X_t.shape)
