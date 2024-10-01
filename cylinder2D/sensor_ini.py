import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys

sys.path.append('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO')
from utils.io import load_loc_of_sensors
from utils.sampler import RandomSampler, UniformSampler, LHSampler
from utils.sampler import ConditionNumberSampler, EnhancedClusteringSampler, ConditionNumberGPUSampler, DeterminantBasedGPUSampler
from utils.sampler import CorrelationClusteringSampler, DeterminantBasedSampler, EfficientConditionNumberSampler, EfficientDeterminantBasedSampler


class Sensor_ini():
    def __init__(self, data, xs, num_obs):
        super(Sensor_ini, self).__init__()
        df_x = pd.read_csv('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/cylinder_xx.csv', header=None, delim_whitespace=False)
        dataset_x = df_x.values
        self.x_ref = dataset_x[:, :][7:119, 0:192]
        df_y = pd.read_csv('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/cylinder_yy.csv', header=None, delim_whitespace=False)
        dataset_y = df_y.values
        self.y_ref = dataset_y[:, :][7:119, 0:192]
        self.num_obs = num_obs
        self.xs = xs
        self.data = data

    def set_init(self, args):
        locations_dic = load_loc_of_sensors(
            'F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/locations_of_sensors_cy.txt',
            num_methods=7)
        key = str(args.num_obs)
        print('\n \n Ini obs method:', args.xgrid_ini_method)
        self.pivots = locations_dic[key][args.xgrid_ini_method]
        self.xgrid_obs_ini_1 = self.xs[self.pivots, :1]
        self.xgrid_obs_ini_2 = self.xs[self.pivots, 1:]
        xgrid_obs_ini = self.xs[self.pivots, :]
        print('obs ini', self.pivots)
        return xgrid_obs_ini

    def set_init_random(self, args):
        sampler = RandomSampler(self.data.T.reshape(-1, 112, 192), num=self.num_obs)
        self.pivots = sampler.sample()
        self.xgrid_obs_ini_1 = self.xs[self.pivots, :1]
        self.xgrid_obs_ini_2 = self.xs[self.pivots, 1:]
        xgrid_obs_ini = self.xs[self.pivots, :]
        return xgrid_obs_ini

    def custom_8_16_ini(self):
        if self.num_obs == 8:
            coords = np.array([[76, 71], [175, 69], [138, 49],
                               [41, 56], [141, 61], [30, 41],
                               [177, 40], [80, 55]])
        elif self.num_obs == 16:
            coords = np.array([[76, 71], [175, 69], [138, 49],
                               [41, 56], [141, 61], [30, 41],
                               [177, 40], [80, 55], [60, 41], [70, 60],
                               [100, 60], [120, 51], [160, 80], [165, 50],
                               [180, 60], [30, 70]])
        else:
            raise ('wrong sensor number')
        coords = np.flip(coords, axis=1)
        self.xgrid_obs_ini_1 = self.x_ref[coords[:, 0], coords[:, 1]].reshape(-1, 1)
        self.xgrid_obs_ini_2 = self.y_ref[coords[:, 0], coords[:, 1]].reshape(-1, 1)

        xgrid_obs_ini = np.concatenate([self.xgrid_obs_ini_1, self.xgrid_obs_ini_2], axis=1)
        return xgrid_obs_ini

    def set_Leverage_score(self, random_seed=12345):
        Xsmall = self.data.T  # n_power x ns
        np.random.seed(random_seed)
        n_snapshots_train, n_pix = Xsmall.shape

        U, s, Q = np.linalg.svd(sklearn.preprocessing.scale(Xsmall, axis=1, with_mean=False, with_std=False, copy=True),
                                full_matrices=False)
        lev_score = np.sum(Q ** 2, axis=0)

        pivots = np.random.choice(range(n_pix), self.num_obs, replace=False, p=lev_score / (n_snapshots_train))
        print('\n Leverage_score x_obs ini', pivots)
        self.xgrid_obs_ini_1 = self.xs[pivots, :1]
        self.xgrid_obs_ini_2 = self.xs[pivots, 1:]
        xgrid_obs_ini = self.xs[pivots, :]
        return xgrid_obs_ini

    def plot_ini(self):
        x_range = self.xs[:, 0].max() - self.xs[:, 0].min()
        y_range = self.xs[:, 1].max() - self.xs[:, 1].min()
        # 设置图像大小比例
        aspect_ratio = x_range / y_range
        # 创建图像，设置大小
        plt.figure(figsize=(6 * aspect_ratio, 6))
        plt.scatter(self.xs[:, :1], self.xs[:, 1:], c=self.data[:, :1], cmap='seismic')
        plt.colorbar()
        # plt.scatter(self.xs[:, 0], self.xs[:, 1])
        plt.scatter(self.xgrid_obs_ini_1, self.xgrid_obs_ini_2, s=100)
        import cmocean
        # plt.contourf(self.xs[:, :1].reshape(112, 192), self.xs[:, 1:].reshape(112, 192),
        #              self.data[:, 99:100].reshape(112, 192), levels=200, cmap='seismic')
        plt.xticks([])
        plt.yticks([])
        # plt.savefig('{}.pdf'.format('./fig/cydinder'), bbox_inches='tight', pad_inches=0)
        plt.show()
