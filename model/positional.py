import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

def PositionalEncoder(image_shape,num_frequency_bands,max_frequencies=None):
    
    *spatial_shape, _ = image_shape
   
    coords = [ torch.linspace(-1, 1, steps=s) for s in spatial_shape ]
    pos = torch.stack(torch.meshgrid(*coords), dim=len(spatial_shape)) 
    
    encodings = []
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    frequencies = [ torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                                              for max_freq in max_frequencies ]
    
    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(pos[..., i:i+1] * frequencies_i[None, ...])

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    enc = torch.cat(encodings, dim=-1)
    enc = rearrange(enc, "... c -> (...) c")

    return enc


def positional_encoding_revise(sensors_locations, max_f, space_bands):
    """
    Compute the positional encoding for a set of sensor locations.

    Parameters:
    sensors_locations (torch.Tensor): A tensor of sensor locations.
    max_f (int): The maximum frequency for the positional encoding.
    space_bands (int): The number of space bands, essentially the length of the encoding.

    Returns:
    torch.Tensor: A tensor containing the positional encodings for the sensor locations.
    """
    # Initialize the positional encoding matrix.
    encoding = torch.zeros(len(sensors_locations), 2 * max_f)

    # Create the frequency vector.
    j = torch.arange(1, max_f + 1)

    # Compute the positional encoding for each sensor location.
    for i, x_i in enumerate(sensors_locations):
        # Compute the positional encoding components.
        encoding[i, 0:max_f] = torch.cos(torch.pi * x_i * j)
        encoding[i, max_f:] = torch.sin(torch.pi * x_i * j)
    # Return the positional encoding with space_bands dimension.
    return encoding[:, :2 * space_bands]

