# cython: language_level=3
import pyximport
pyximport.install()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import copy
import sys
sys.path.append('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO')
# from model.neural_operator import DeepONet
from utils.misc import prep_experiment_dir
from utils.ImplicitLayer import ImplicitLayer
from utils.default_options import parses
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
f_npy = lambda x: x.cpu().detach().numpy()
f_tensor = lambda x: torch.tensor(x).float().to(device)

class Solver_basic(object):
    def __init__(self, args):
        # self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.exp = 'recon_cy_2D_{}_{}_obs{}_test{}'.format(args.model, args.xgrid_ini_method, args.num_obs, args.test)
        prep_experiment_dir(args)
        self.run_path = args.fig_path + f'/run_out.txt'
        x1_min, x2_min = np.min(args.xs[:, 0]), np.min(args.xs[:, 1])
        x1_max, x2_max = np.max(args.xs[:, 0]), np.max(args.xs[:, 1])
        self.xgrid_min = f_tensor([x1_min, x2_min])  # 2d [0.0,0.0]
        self.xgrid_max = f_tensor([x1_max, x2_max])
        self.x_clamp_range = (self.xgrid_max - self.xgrid_min) / 2
        self.xs = args.xs
        self.criterion = F.l1_loss
        # self.criterion_test = F.l1_loss
        self.criterion_l2 = torch.nn.MSELoss()
        # self.criterion = torch.nn.MSELoss()
        self.criterion_test = torch.nn.MSELoss()
        # set input
        self.y_exact = f_tensor(args.y_exact_flatten)
        # n_train, n_val = 0.7, 0.15
        n_train, n_val, n_test = 0.7, 0.15, 0.15
        self.y_train_exact, self.y_val_exact = (self.y_exact[:int(args.n_power * n_train),:],
                                                self.y_exact[int(args.n_power * n_train):int(args.n_power * (n_train+n_val)),:])
        self.y_test_exact = self.y_exact[int(args.n_power * (n_train+n_val)):int(args.n_power * (n_train+n_val+n_test)),:]  # (n_test x ns)
        self.batchsize = 16
        # self.batchsize = 750
        self.std = 1.0
        self.y_mean_denom = self.criterion_l2(self.y_test_exact, torch.zeros_like(self.y_test_exact)).to(self.device)

        # early-stopping
        self.best_train_loss = float('inf')
        self.patience = 100  # 设置耐心阈值
        self.patience_counter = 0  # 初始化耐心计数器
        self.min_delta = 1e-5  # 设置性能改善的最小阈值

        # set draft param
        self.best_val, self.best_test, self.worse_update = float('inf'), float('inf'), 0
        self.best_rec, self.best_xgrid_obs_log = [], []
        self.xgrid_obs_log = []
        self.best_net_para = None
        self.net = None

        self.xgrid_obs_ini = f_tensor(args.xgrid_obs_ini)
        self.best_xgrid_obs = self.xgrid_obs_ini
        self.dydx12_val = None
        # global momentum
        # momentum = torch.zeros_like(self.xgrid_obs_ini)
        # 初始化 Adam 的状态变量
        self.exp_avg = torch.zeros_like(self.xgrid_obs_ini)
        self.exp_avg_sq = torch.zeros_like(self.xgrid_obs_ini)
        self.Step = 0
        print('Model: {},Test:{}'.format(args.model, args.test))
        with open(self.run_path, 'a') as f:
            f.write('Model: {},Test:{} \n'.format(args.model, args.test))
        self.best_loss, self.patience, self.patience_counter = float('inf'), 30, 0
        self.fig_path = args.fig_path
        self.args = args

    def RBFInteploation_train(self, xgrid_obs):
        layer_train = ImplicitLayer(f_npy(xgrid_obs), self.xs, f_npy(self.y_train_exact.T), False)
        _, _, y_obs_train = layer_train()
        return y_obs_train
    def RBFInteploation_val(self, xgrid_obs):
        layer_val = ImplicitLayer(f_npy(xgrid_obs), self.xs, f_npy(self.y_val_exact.T), True)
        dydx1_val, dydx2_val, y_obs_val = layer_val()
        return dydx1_val, dydx2_val, y_obs_val
    def RBFInteploation_test(self, xgrid_obs):
        layer_test = ImplicitLayer(f_npy(xgrid_obs), self.xs, f_npy(self.y_test_exact.T), False)
        _, _, y_obs_test = layer_test()
        return y_obs_test

    def RBFInteploation_train_s(self, xgrid_obs): # senseiver模型
        layer_train = ImplicitLayer(f_npy(xgrid_obs), self.xs, f_npy(self.y_train_exact.T), False)
        _, _, y_obs_train = layer_train()
        return y_obs_train
    def RBFInteploation_val_s(self, xgrid_obs): # senseiver模型
        layer_val = ImplicitLayer(f_npy(xgrid_obs), self.xs, f_npy(self.y_val_exact.T), False)
        _, _, y_obs_val = layer_val()
        return y_obs_val
    def RBFInteploation_test_s(self, xgrid_obs): # senseiver模型
        layer_test = ImplicitLayer(f_npy(xgrid_obs), self.xs, f_npy(self.y_test_exact.T), False)
        _, _, y_obs_test = layer_test()
        return y_obs_test

    def obs_scope(self, args, xgrid_obs, dim=1):
        xgrid_obs_clamp = torch.clamp(xgrid_obs, self.xgrid_min, self.xgrid_max)
        for i in range(dim):
            xgrid_obs_clamp[:, i] = torch.where(xgrid_obs[:, i] < self.xgrid_min[i],
                                                torch.rand(args.num_obs).to(self.device) * (
                                                        self.xgrid_min[i] - self.xgrid_max[i]) + self.xgrid_max[i],
                                                xgrid_obs_clamp[:, i])
            xgrid_obs_clamp[:, i] = torch.where(xgrid_obs[:, i] > self.xgrid_max[i],
                                                torch.rand(args.num_obs).to(self.device) * (
                                                        self.xgrid_max[i] - self.xgrid_min[i]) + self.xgrid_min[i],
                                                xgrid_obs_clamp[:, i])
        return xgrid_obs_clamp
    def set_dataset(self, inputs, outputs, shuffle=True):
        dataset = TensorDataset(inputs, outputs)
        loader = DataLoader(dataset, batch_size=self.batchsize, shuffle=shuffle)
        return loader

    def set_dataset_coff(self, inputs, outputs, labels, shuffle=True):
        dataset = TensorDataset(inputs, outputs, labels)
        loader = DataLoader(dataset, batch_size=self.batchsize, shuffle=shuffle)
        return loader

    def reset_weights_bias(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:  # 如果参数是权重矩阵
                torch.nn.init.xavier_normal_(param)
            if 'bias' in name:  # 如果参数是偏置向量
                torch.nn.init.zeros_(param)


    def optim_x_adam_revise(self, xgrid_obs, grad_x_grid_obs, exp_avg, exp_avg_sq, step, lr=0.5, beta1=0.9, beta2=0.999,
                     eps=1e-8):
        # 执行 Adam 更新
        step += 1
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad_x_grid_obs
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad_x_grid_obs ** 2

        # 偏差校正
        corrected_exp_avg = exp_avg / (1 - beta1 ** step)
        corrected_exp_avg_sq = exp_avg_sq / (1 - beta2 ** step)

        # 更新 xgrid_obs
        denom = corrected_exp_avg_sq.sqrt().add(eps)
        xgrid_obs_ini = xgrid_obs
        xgrid_obs = xgrid_obs - lr * corrected_exp_avg / denom
        # 边界点处理
        # xgrid_obs_clamp = torch.clamp(xgrid_obs, self.xgrid_min, self.xgrid_max)
        for i in range(2):
            for j in range(xgrid_obs.shape[0]):
                xgrid_obs[j, i] = torch.where(xgrid_obs[j, i] < self.xgrid_min[i],
                                                    xgrid_obs_ini[j, i] + 3*lr * (corrected_exp_avg / denom)[j, i], xgrid_obs[j, i])
                xgrid_obs[j, i] = torch.where(xgrid_obs[j, i] > self.xgrid_max[i],
                                                    xgrid_obs_ini[j, i] + 3*lr * (corrected_exp_avg / denom)[j, i], xgrid_obs[j, i])
        xgrid_obs = torch.clamp(xgrid_obs, self.xgrid_min, self.xgrid_max)
        with open(self.run_path, 'a') as f:
            # f.write('ini:{} \n'.format(xgrid_obs_ini))
            f.write('delta:{} \n'.format(lr * corrected_exp_avg / denom))
            # f.write('update:{} \n'.format(xgrid_obs))
            # f.write('dydx:{}\n'.format(self.dydx12_val))
            f.write('grad_x_grid_obs:{} \n'.format(grad_x_grid_obs))
        return xgrid_obs, exp_avg, exp_avg_sq, step

    def record_best(self, val_loss, test_loss, xgrid_obs, xgrid_obs_epoch):
        # self.best_net_para = copy.deepcopy(self.net.state_dict())
        self.best_net_para = copy.deepcopy(self.net)
        self.best_xgrid_obs = copy.deepcopy(xgrid_obs)
        self.best_rec.append([val_loss, test_loss, val_loss])
        self.best_val = val_loss
        self.best_test = test_loss
        self.best_xgrid_obs_log.append([f_npy(xgrid_obs), val_loss])
        self.worse_update = 0
        torch.save(self.best_net_para, self.args.fig_path + f'/model/{self.args.model}_best_model.pth')
        torch.save(self.best_xgrid_obs, self.args.fig_path + f'/model/{self.args.model}_best_obs.pth')


    def plot_train_loss(self, args, loss_history, val_loss, test_loss, xgrid_obs_epoch):
        plt.plot(loss_history)
        plt.legend(('Train Loss', 'Val Loss'))
        plt.suptitle(f"{xgrid_obs_epoch}, val {val_loss}, test {test_loss}")
        plt.yscale('log')
        plt.ylim(1e-2, 1e0)
        plt.savefig(args.fig_path + f'/loss_log/pred_{xgrid_obs_epoch}.png')
        plt.close()

    def plot_obs(self, args, xgrid_obs, u_pred_plot, u_error_plot, u_val_plot, i_plot, val_loss, test_loss, plot_save):
        fig, ax = plt.subplots(1, 2, figsize=(19, 6))
        fig0 = ax[0].scatter(self.xs[:, :1], self.xs[:, 1:], c=u_pred_plot, cmap='seismic')
        ax[0].scatter(f_npy(xgrid_obs[:, :1]), f_npy(xgrid_obs[:, 1:]), c='black', s=100)
        cbar0 = plt.colorbar(fig0, ax=ax[0])
        fig1 = ax[1].scatter(self.xs[:, :1], self.xs[:, 1:], c=u_error_plot, cmap='seismic')
        ax[1].scatter(f_npy(xgrid_obs[:, :1]), f_npy(xgrid_obs[:, 1:]), c='black', s=100)
        cbar1 = plt.colorbar(fig1, ax=ax[1])
        plt.suptitle(f"{i_plot}, val {val_loss}, test {test_loss}")
        if plot_save == 'best':
            plt.savefig(args.fig_path + f'/best/pred_{i_plot}.png')
            plt.close()
            # np.savez(args.fig_path + f'/model/pred_error_{i_plot}.npz', DSPO_pred=u_pred_plot,error=u_error_plot, true= u_val_plot)
        else:
            plt.savefig(args.fig_path + f'/obs_log/pred_{i_plot}.png')
            plt.close()

    def plot_optimizer_loss(self, args, best_rec):
        plt.cla()
        plt.plot(best_rec)
        plt.yscale('log')
        # plt.ylim(1e-9, 1e-3)
        plt.legend(('Val', 'Test', 'val_log'))
        plt.savefig(args.fig_path + f'/best_val.png')
        plt.show()

    def save_error_log(self, args, best_net_para, best_xgrid_obs, best_xgrid_obs_log, best_rec):
        torch.save(best_net_para, args.fig_path + f'/model/best_model.pth')
        torch.save(best_xgrid_obs, args.fig_path + f'/model/best_obs.pth')
        torch.save(best_xgrid_obs_log, args.fig_path + f'/model/best_obs_log.pth')
        torch.save(best_rec, args.fig_path + f'/model/best_rec_log.pth')


    def optim_x_MSGD(self, xgrid_obs, grad_x_grid_obs, momentum, grad_scale=1e2, momentum_factor=0.9,
                     learning_rate=0.1):
        print('momentum', momentum)
        grad_scale = grad_scale
        scaled_gradients = grad_scale * grad_x_grid_obs
        with torch.no_grad():
            momentum = momentum_factor * momentum + scaled_gradients
            # xgrid_obs -= learning_rate * momentum
            xgrid_obs -= torch.clamp(learning_rate * momentum, -0.2 * self.x_clamp_range, 0.2 * self.x_clamp_range)
        print('x_grad_update: \n', f_npy(learning_rate * momentum))
        return xgrid_obs, momentum

    def optim_x_adam(self, xgrid_obs, grad_x_grid_obs, exp_avg, exp_avg_sq, step, lr=0.5, beta1=0.9, beta2=0.999,
                     eps=1e-8):
        # 执行 Adam 更新
        step += 1
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad_x_grid_obs
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad_x_grid_obs ** 2

        # 偏差校正
        corrected_exp_avg = exp_avg / (1 - beta1 ** step)
        corrected_exp_avg_sq = exp_avg_sq / (1 - beta2 ** step)

        # 更新 xgrid_obs
        denom = corrected_exp_avg_sq.sqrt().add(eps)
        xgrid_obs = xgrid_obs - lr * corrected_exp_avg / denom
        print('delta', lr * corrected_exp_avg / denom)
        return xgrid_obs, exp_avg, exp_avg_sq, step
