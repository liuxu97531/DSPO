import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import datetime
import sys
sys.path.append('..')
from model.positional import PositionalEncoder
from model.positional import positional_encoding_revise
from torch.utils.data import DataLoader, Dataset
import pandas as pd
class senseiver_loader(Dataset):
    def __init__(self,   xs_obs, ys_obs, y_obs, y_data_square, data_config):
        xs_obs, ys_obs       = xs_obs.cpu(), ys_obs.cpu()
        self.indexed_sensors = y_obs.unsqueeze(2).cpu()
        self.data            =  y_data_square.permute(0, 2, 3, 1).cpu()      # (nx, xs, ys, 1)
        self.training_frames = y_data_square.size(0)
        self.batch_frames    = data_config['batch_frames'] 
        self.batch_pixels    = data_config['batch_pixels']
        pix_total_num = data_config['pix_total_num']
        num_batches = int(pix_total_num*self.training_frames/(
                                            self.batch_frames*self.batch_pixels))
        assert num_batches > 0
        print(f'{num_batches} Batches of data per epoch\n')
        data_config['num_batches'] = num_batches
        self.num_batches = num_batches
        self.train_ind = torch.arange(0, self.training_frames)

        # sine-cosine positional encodings
        self.sensor_positions_xs = positional_encoding_revise(xs_obs, 200, data_config['space_bands'])
        self.sensor_positions_ys = positional_encoding_revise(ys_obs, 200, data_config['space_bands'])
        self.sensor_positions = torch.cat([self.sensor_positions_xs,self.sensor_positions_ys],dim=1)
        self.sensor_positions = self.sensor_positions[None,].repeat_interleave(
                                                    self.batch_frames, axis=0)
        self.pos_encodings = PositionalEncoder(self.data.shape[1:], data_config['space_bands'])
        # get non-zero pixels
        self.pix_avail = self.data.flatten(start_dim=1, end_dim=-2)[0,:,0].nonzero()[:,0] # 21504


    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        frames = self.train_ind[ torch.randperm(self.training_frames) ][:self.batch_frames]
        pixels = self.pix_avail[ torch.randperm(*self.pix_avail.shape) ][:self.batch_pixels] # 注意力机制，只选择2048个像素点
        # pixels = self.pix_avail
        sensor_values = self.indexed_sensors[frames,]
        sensor_values = torch.cat([sensor_values,self.sensor_positions], axis=-1)
        coords = self.pos_encodings[pixels,][None,]
        coords = coords.repeat_interleave(self.batch_frames, axis=0)
        field_values = self.data.flatten(start_dim=1, end_dim=-2)[frames,][:,pixels,]
        return sensor_values, coords, field_values


def senseiver_loader_val(xs_obs, ys_obs, y_obs, y_data_square, data_config):
    xs_obs, ys_obs = xs_obs.cpu(), ys_obs.cpu()
    indexed_sensors = y_obs.unsqueeze(2).cpu()
    data = y_data_square.permute(0, 2, 3, 1).cpu()  # (nx, xs, ys, 1)
    training_frames = y_data_square.size(0)
    batch_frames = data_config['batch_frames']
    batch_pixels = data_config['pix_total_num']
    pix_total_num = data_config['pix_total_num']
    num_batches = int(pix_total_num * training_frames / (
            batch_frames * batch_pixels))
    assert num_batches > 0
    print(f'{num_batches} Batches of data per epoch\n')
    data_config['num_batches'] = num_batches
    num_batches = num_batches
    train_ind = torch.arange(0, training_frames)

    # sine-cosine positional encodings
    sensor_positions_xs = positional_encoding_revise(xs_obs, 200, data_config['space_bands'])
    sensor_positions_ys = positional_encoding_revise(ys_obs, 200, data_config['space_bands'])
    sensor_positions = torch.cat([sensor_positions_xs, sensor_positions_ys], dim=1)
    # sensor_positions = sensor_positions[None,].repeat_interleave(
    #     batch_frames, axis=0)
    sensor_positions = sensor_positions[None,].repeat_interleave(
        training_frames, axis=0)
    pos_encodings = PositionalEncoder(data.shape[1:], data_config['space_bands'])
    # get non-zero pixels
    pix_avail = data.flatten(start_dim=1, end_dim=-2)[0, :, 0].nonzero()[:, 0]  # 21504

    frames = train_ind[:batch_frames]
    # pixels = self.pix_avail[torch.randperm(*self.pix_avail.shape)][:self.batch_pixels]  # 注意力机制，只选择2048个像素点
    pixels = pix_avail
    # sensor_values = indexed_sensors[frames]
    sensor_values = indexed_sensors
    sensor_values = torch.cat([sensor_values, sensor_positions], axis=-1)

    coords = pos_encodings[pixels,][None,]
    coords = coords.repeat_interleave(batch_frames, axis=0)

    # field_values = Data_total.flatten(start_dim=1, end_dim=-2)[frames,][:, pixels, ]
    field_values = data.flatten(start_dim=1, end_dim=-2)[:, pixels, ]
    return sensor_values, coords, field_values, num_batches


class senseiver_loader_test(Dataset):
    def __init__(self, xs_obs, ys_obs, y_obs, y_data_square, data_config):
        xs_obs, ys_obs = xs_obs.cpu(), ys_obs.cpu()
        self.indexed_sensors = y_obs.unsqueeze(2).cpu()
        self.data = y_data_square.permute(0, 2, 3, 1).cpu()  # (nx, xs, ys, 1)
        self.training_frames = y_data_square.size(0)
        self.batch_frames = data_config['batch_frames']
        self.batch_pixels = data_config['pix_total_num']
        pix_total_num = data_config['pix_total_num']
        num_batches = int(pix_total_num * self.training_frames / (
                self.batch_frames * self.batch_pixels))
        assert num_batches > 0
        print(f'{num_batches} Batches of data per epoch\n')
        data_config['num_batches'] = num_batches
        self.num_batches = num_batches
        self.train_ind = torch.arange(0, self.training_frames)

        # sine-cosine positional encodings
        self.sensor_positions_xs = positional_encoding_revise(xs_obs, 200, data_config['space_bands'])
        self.sensor_positions_ys = positional_encoding_revise(ys_obs, 200, data_config['space_bands'])
        self.sensor_positions = torch.cat([self.sensor_positions_xs, self.sensor_positions_ys], dim=1)
        self.sensor_positions = self.sensor_positions[None,].repeat_interleave(
            self.batch_frames, axis=0)
        self.pos_encodings = PositionalEncoder(self.data.shape[1:], data_config['space_bands'])
        # get non-zero pixels
        self.pix_avail = self.data.flatten(start_dim=1, end_dim=-2)[0, :, 0].nonzero()[:, 0]  # 21504

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        frames = self.train_ind[torch.randperm(self.training_frames)][:self.batch_frames]
        # pixels = self.pix_avail[torch.randperm(*self.pix_avail.shape)][:self.batch_pixels]  # 注意力机制，只选择2048个像素点
        pixels = self.pix_avail
        sensor_values = self.indexed_sensors[frames,]
        sensor_values = torch.cat([sensor_values, self.sensor_positions], axis=-1)
        coords = self.pos_encodings[pixels,][None,]
        coords = coords.repeat_interleave(self.batch_frames, axis=0)
        field_values = self.data.flatten(start_dim=1, end_dim=-2)[frames,][:, pixels, ]
        return sensor_values, coords, field_values
