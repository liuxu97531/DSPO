import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from model.rff import PositionalEncoding


class DeterministicEncoder(nn.Module):
    def __init__(self, sizes):
        super(DeterministicEncoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, context_x, context_y):
        """
        Encode training set as one vector representation

        Args:
            context_x:  batch_size x set_size x feature_dim
            context_y:  batch_size x set_size x 2

        Returns:
            representation:
        """
        encoder_input = torch.cat((context_x, context_y), dim=-1)
        batch_size, set_size, filter_size = encoder_input.shape
        x = encoder_input.view(batch_size * set_size, -1)
        for i, linear in enumerate(self.linears[:-1]):
            x = F.gelu(linear(x))
        x = self.linears[-1](x)
        x = x.view(batch_size, set_size, -1)
        representation = x.mean(dim=1)
        return representation


class DeterministicDecoder(nn.Module):
    def __init__(self, sizes):
        super(DeterministicDecoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, representation, target_x):
        """
        Take representation representation of current training set, and a target input x,
        return the probability of x being positive

        Args:
            representation: batch_size x representation_size
            target_x: batch_size x set_size x d
        """
        batch_size, set_size, d = target_x.shape
        representation = representation.unsqueeze(1).repeat([1, set_size, 1])
        input = torch.cat((representation, target_x), dim=-1)
        x = input.view(batch_size * set_size, -1)
        for i, linear in enumerate(self.linears[:-1]):
            x = F.gelu(linear(x))
        x = self.linears[-1](x)
        out = x.view(batch_size, set_size, -1)
        # mu, log_sigma = torch.split(out, 2, dim=-2)
        # sigma = 0.2 + 0.9 * torch.nn.functional.softplus(log_sigma)
        # dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        # return dist, mu, sigma
        return out


def input_mapping(x, B):
    if B is None:
        return x
    else:
        B = B.to(x.device)
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ConditionalNeuralProcess(nn.Module):
    def __init__(self, encoder_sizes, decoder_sizes, fourier_feature=False):
        super(ConditionalNeuralProcess, self).__init__()
        self._encoder = DeterministicEncoder(encoder_sizes)
        self._decoder = DeterministicDecoder(decoder_sizes)
        self.fourier_feature = fourier_feature
        if fourier_feature:
            self.position_encoding = PositionalEncoding(sigma=1.0, m=32)

    def forward(self, x_context, y_context, x_target, y_target=None):
        if self.fourier_feature:
            # x_context = input_mapping(x_context, self.B_gauss)
            x_target = self.position_encoding(x_target)
        representation = self._encoder(x_context, y_context)
        # dist, mu, sigma = self._decoder(representation, x_target)
        #
        # log_p = None if y_target is None else dist.log_prob(y_target)
        # return log_p, mu, sigma
        return self._decoder(representation, x_target)


if __name__ == '__main__':
    cnp = ConditionalNeuralProcess(encoder_sizes=[3, 256, 512, 512, 256],
                                   decoder_sizes=[64 * 2 + 256, 256, 256, 256, 1],
                                   fourier_feature=True)
    x_context = torch.randn(2, 4000, 2)
    y_context = torch.randn(2, 4000, 1)
    x_target = torch.randn(2, 40000, 2)
    with torch.no_grad():
        y_target = cnp(x_context, y_context, x_target)
    print(y_target.shape)
