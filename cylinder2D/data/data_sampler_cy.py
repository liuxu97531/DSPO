import matplotlib.pyplot as plt
import h5py
import pickle
import cmocean
import pandas as pd
import torch
import numpy as np
import sys
sys.path.append('../..')
from utils.sampler import RandomSampler, UniformSampler, LHSampler
from utils.sampler import ConditionNumberSampler, EnhancedClusteringSampler, ConditionNumberGPUSampler, DeterminantBasedGPUSampler
from utils.sampler import CorrelationClusteringSampler, DeterminantBasedSampler, EfficientConditionNumberSampler, EfficientDeterminantBasedSampler

def Data_PSO_cy():
    num_obs = 32
    # n_data = 3500
    n_data = 5000
    df = pd.read_csv('./cylinder_xx.csv', header=None, delim_whitespace=False)
    dataset = df.values
    x = dataset[:, :]
    x_ref = x[7:119, 0:192].reshape(-1, 1)  # (112, 192) # 非圆柱内坐标
    print(x.shape)  # (125, 200)
    df = pd.read_csv('./cylinder_yy.csv', header=None, delim_whitespace=False)
    dataset = df.values
    y = dataset[:, :]
    y_ref = y[7:119, 0:192].reshape(-1, 1)  # 非圆柱内坐标
    filename = "./Cy_Taira.pickle"
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        omg_box = np.squeeze(obj, axis=3).reshape(5000, -1).T
        omg_box1 = obj.transpose(0, 3, 1, 2)
    print(omg_box.shape)  # (21504,5000)
    y_exact_square = omg_box1
    xs_1, xs_2, y_exact_flatten = x_ref, y_ref, omg_box.T
    coor = np.concatenate([xs_1, xs_2], axis=1)
    data_u = y_exact_square[:n_data,:,:,:]
    sampler = RandomSampler(np.squeeze(data_u, axis=1), num=num_obs)
    # sampler = UniformSampler(np.squeeze(data_u, axis=1), num=num_obs)
    # sampler = LHSampler(np.squeeze(data_u, axis=1), num=num_obs)
    # sampler = EfficientConditionNumberSampler(data_u, coor=coor, num=num_obs, n_components=30)
    # sampler = EfficientDeterminantBasedSampler(data_u, coor=coor, num=num_obs, n_components=30)
    # sampler = EnhancedClusteringSampler(data_u, num=num_obs, coor=coor, n_clusters=100)
    # sampler = CorrelationClusteringSampler(data_u, num=num_obs, coor=coor, n_clusters=100)

    locations = sampler.sample()

    # Plotting
    print('obs idx:', locations)
    data = np.zeros((data_u.shape[-2], data_u.shape[-1])).flatten()
    data[locations] = 1
    plt.subplot(121)
    plt.imshow(data.reshape(data_u.shape[-2], data_u.shape[-1]))
    plt.subplot(122)
    plt.imshow(data_u[0, 0, :, :])
    plt.show()

    xgrid_obs_ini = coor[locations,:]
    x_range = coor[:, 0].max() - coor[:, 0].min()
    y_range = coor[:, 1].max() - coor[:, 1].min()
    # 设置图像大小比例
    aspect_ratio = x_range / y_range
    # 创建图像，设置大小
    data = np.squeeze(data_u, axis=1).reshape(n_data, -1)
    plt.figure(figsize=(6 * aspect_ratio, 6))
    plt.contourf(coor[:, :1].reshape(112, 192), coor[:, 1:].reshape(112, 192),data[:1, :].reshape(112, 192),
                 levels=200, cmap=cmocean.cm.balance)
    plt.colorbar()
    # plt.scatter(coor[:, :1], coor[:, 1:], c='b', s=600)
    plt.scatter(xgrid_obs_ini[:, :1], xgrid_obs_ini[:, 1:], c='r', s=600)
    plt.show()
    colors =['#99ABB9', '#94BCE4', '#C7C7F1','#91ABDF','#ACACEA','#96C1F4','#6699FF', '#97BACF', '#0099CC']
    # plt.figure(figsize=(6 * 5, 6))c
    j = 8
    for i in range(min(9, len(locations))):
        # plt.subplot(3, 3, i + 1)
        plt.plot(data_u.reshape(data_u.shape[0], -1)[:, locations[i]],color=colors[j])
        plt.title('Obs')
        plt.savefig(f'./obs_plot/obs_{j}_{i}')
        plt.show()

if __name__ == '__main__':
    Data_PSO_cy()