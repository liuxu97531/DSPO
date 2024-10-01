import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import sys
sys.path.append('..')

def CylinderDataset(index):
    df = pd.read_csv('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/cylinder_xx.csv', header=None, delim_whitespace=False)
    dataset = df.values
    x = dataset[:, :]
    x_ref = x[7:119, 0:192].reshape(-1, 1)  # (112, 192) # 非圆柱内坐标
    print(x.shape)  # (125, 200)
    df = pd.read_csv('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/cylinder_yy.csv', header=None, delim_whitespace=False)
    dataset = df.values
    y = dataset[:, :]
    y_ref = y[7:119, 0:192].reshape(-1, 1)  # 非圆柱内坐标
    filename = "F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/Cy_Taira.pickle"
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        omg_box = np.squeeze(obj, axis=3).reshape(5000, -1).T
        omg_box1 = obj.transpose(0, 3, 1, 2)
    print(omg_box.shape)  # (21504,5000)
    y_exact_square = omg_box1
    xs_1, xs_2, y_exact_flatten = x_ref, y_ref, omg_box.T
    xs = np.concatenate([xs_1, xs_2], axis=1)
    return xs, y_exact_flatten[index, :], y_exact_square[index, :, :, :]

class CylinderPodDataset():
    def __init__(self, args, pod_index, index, n_components=20, mean=0, std=1):
        """
        圆柱绕流数据集：通过POD对物理场进行降维，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(CylinderPodDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/Cy_Taira.pickle', 'rb')
        self.pca_data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[pod_index, :, :, :]
        self.pca_data = (self.pca_data - mean) / std
        df.close()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)
        df = open('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/Cy_Taira.pickle', 'rb')
        self.data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[:, :, :, :]
        self.data = (self.data - mean) / std
        df.close()
        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        self.size = self.data.shape[-3:]

        df = pd.read_csv('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/cylinder_xx.csv', header=None, delim_whitespace=False)
        dataset = df.values
        x = dataset[:, :]
        x_ref = x[7:119, 0:192].reshape(-1, 1)  # (112, 192) # 非圆柱内坐标
        print(x.shape)  # (125, 200)
        df = pd.read_csv('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D/data/cylinder_yy.csv', header=None, delim_whitespace=False)
        dataset = df.values
        y = dataset[:, :]
        y_ref = y[7:119, 0:192].reshape(-1, 1)  # 非圆柱内坐标
        # filename = "Data_total/Cy_Taira.pickle"
        # with open(filename, 'rb') as f:
        #     obj = pickle.load(f)
        #     omg_box = np.squeeze(obj, axis=3).reshape(5000, -1).T
        # print(omg_box.shape)  # (5000, 21504)
        xs_1, xs_2 = x_ref, y_ref
        xs = np.concatenate([xs_1, xs_2], axis=1)
        args.xs = xs
        args.pod_coff = self.coff[index, :]
        args.y_exact_flatten = np.squeeze(self.data.numpy(), axis=1).reshape(5000, -1)[index, :]  # n_power x ns

    def inverse_transform(self, coff):
        inverse_coff = coff * (self.max - self.min) + self.min
        return self.pca.inverse_transform(inverse_coff)



if __name__ == '__main__':
    CylinderDataset(index=[i for i in range(5000)])