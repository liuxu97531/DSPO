# cython: language_level=3
import pyximport
pyximport.install()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import trange
import numpy as np
import copy
import torch
from torch import nn, optim
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
sys.path.append('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D')
from sensor_ini import Sensor_ini
from CylinderDataset import CylinderDataset
from DSPO_optimizer_cy import Solver_basic
from senseiver_parser import parse_args
from sensiver_dataloaders import senseiver_loader, senseiver_loader_val, senseiver_loader_test
sys.path.append('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO')
from model.senseiver import Senseiver
from utils.utils import setup_seed
from utils.default_options import parses

f_npy = lambda x: x.cpu().detach().numpy()
f_tensor = lambda x: torch.tensor(x).float().to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class Solver_rec(Solver_basic):
    def __init__(self):
        super().__init__(args)
        self.std = 11.0960
        self.min_delta = 1e-8
        self.y_exact_square = f_tensor(args.y_exact_square)
        n_train, n_val, n_test = 0.7, 0.15, 0.15
        self.y_train_exact_square, self.y_val_exact_square = (
            self.y_exact_square[:int(args.n_power * n_train), :, :, :],
            self.y_exact_square[int(args.n_power * n_train):int(args.n_power * (n_train + n_val)), :, :, :])
        self.y_test_exact_square = self.y_exact_square[int(args.n_power * (n_train + n_val)):, :, :, :]  # (n_test x ns)

        self.data_config, encoder_config, decoder_config = parse_args()
        self.net = Senseiver(
            **encoder_config,
            **decoder_config,
            **self.data_config
        ).to(self.device)

        self.xgrid_obs_ini = f_tensor(args.xgrid_obs_ini)

    def train(self, xobs_optimizer):
        args.epochs, iter_obs_update = 1000, 30
        # args.epochs, iter_obs_update = 1, 3
        xgrid_obs = self.xgrid_obs_ini
        if not xobs_optimizer:
            iter_obs_update = 1
        for xgrid_obs_epoch in range(iter_obs_update):
            self.best_train_loss = float('inf')
            y_obs_train = self.RBFInteploation_train(xgrid_obs)
            train_loader = DataLoader(
                senseiver_loader(xgrid_obs[:, :1], xgrid_obs[:, 1:], y_obs_train, self.y_train_exact_square,
                                 self.data_config),
                batch_size=None,
                pin_memory=True,
                shuffle=True,
                num_workers=1
                )
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
            loss_history = []
            pbar = trange(args.epochs, desc='Epochs')
            for epoch in pbar:
                self.net.train()
                train_loss, train_num = 0., 0.
                for batch_idx, (inputs, coords, targets) in enumerate(train_loader):
                    inputs, coords, targets = inputs.to(self.device), coords.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    pre = self.net(inputs, coords)
                    # loss = self.criterion(pre, targets)
                    loss = self.criterion_l2(pre, targets)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.detach().item()
                    # train_loss += loss.item() * inputs.shape[0]
                    # train_num += inputs.shape[0]
                # train_loss = train_loss / train_num
                train_loss = train_loss / len(train_loader)
                # self.scheduler.step()
                loss_history.append([train_loss])
                pbar.set_postfix({'Train Loss': train_loss})

                if self.best_train_loss - train_loss > self.min_delta:
                    self.best_train_loss = train_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                if self.patience_counter > self.patience:
                    print("Early stopping triggered based on training set performance.")
                    break

            self.xgrid_obs_test = copy.deepcopy(xgrid_obs)
            test_loss = self.Predict(loading=False)

            val_loss, val_num, grad_x1_obs_val, grad_x2_obs_val = 0., 0., [], []
            y_obs_val = self.RBFInteploation_val_s(xgrid_obs)
            xs_obs, ys_obs = xgrid_obs[:, :1].requires_grad_(True), xgrid_obs[:, 1:].requires_grad_(True)
            inputs_val, coords_val, targets_val, batch_idx_val = senseiver_loader_val(xs_obs, ys_obs, y_obs_val,
                                                                                      self.y_val_exact_square,
                                                                                      self.data_config)
            batch_frames = self.data_config['batch_frames']
            for idx in range(batch_idx_val):
                inputs, coords, targets = (inputs_val[idx * (batch_frames):(idx + 1) * (batch_frames), :].to(self.device),
                                           coords_val.to(self.device),
                                           targets_val[idx * (batch_frames):(idx + 1) * (batch_frames), :].to(self.device))
                pre = self.net(inputs, coords)
                loss = self.criterion_l2(pre, targets)
                val_loss += loss.item()
                # val_loss += loss.item() * inputs.shape[0]
                val_num += inputs.shape[0]
                grad_x1_obs_batch = grad(loss, xs_obs, create_graph=False, retain_graph=True)[0]
                grad_x2_obs_batch = grad(loss, ys_obs, create_graph=False, retain_graph=True)[0]
                grad_x1_obs_val.append(grad_x1_obs_batch)
                grad_x2_obs_val.append(grad_x2_obs_batch)
            grad_x1_grid_obs = torch.cat(grad_x1_obs_val, dim=1).sum(dim=1) / val_num
            grad_x2_grid_obs = torch.cat(grad_x2_obs_val, dim=1).sum(dim=1) / val_num
            # val_loss = val_loss / val_num
            val_loss = val_loss / batch_idx_val

            # self.plot_obs_update_senseiver(pre, targets, xgrid_obs, xgrid_obs_epoch, val_loss, test_loss, 'obs_log')
            # self.plot_train_loss(args, loss_history, val_loss, test_loss, xgrid_obs_epoch)

            if val_loss <= self.best_val:
                self.record_best(val_loss, test_loss, xgrid_obs, xgrid_obs_epoch)
                # self.plot_obs_update_senseiver(pre, targets, xgrid_obs, xgrid_obs_epoch, val_loss, test_loss, 'best')
            else:
                self.worse_update += 1
                self.best_rec.append([self.best_val, self.best_test, val_loss])

                # print('worse_update', self.worse_update)
            with open(self.run_path, 'a') as f:
                f.write(f'\n\n obs_updata {xgrid_obs_epoch}: Train Loss: {train_loss}, Val Loss: {val_loss} \n')
                f.write(f'worse_update: {self.worse_update} \n')
                f.write(f'test loss: {test_loss}\n')

            grad_x_grid_obs = torch.cat([grad_x1_grid_obs.reshape(-1,1), grad_x2_grid_obs.reshape(-1,1)], dim=1)
            xgrid_obs, self.exp_avg, self.exp_avg_sq, self.Step = self.optim_x_adam_revise(xgrid_obs, grad_x_grid_obs,
                                                                                           self.exp_avg,
                                                                                           self.exp_avg_sq,
                                                                                           self.Step, lr=0.25)
            if self.worse_update >= 6:
                self.net = self.best_net_para
                xgrid_obs = self.best_xgrid_obs
                self.worse_update = 0

        # self.plot_optimizer_loss(args, self.best_rec)
        self.save_error_log(args, self.best_net_para, self.best_xgrid_obs, self.best_xgrid_obs_log, self.best_rec)

    def Predict(self, loading=True):
        if loading:
            self.net = torch.load(args.fig_path + f'/model/best_model.pth')
            self.best_xgrid_obs = torch.load(args.fig_path + f'/model/best_obs.pth')
            y_obs_test = self.RBFInteploation_test_s(self.best_xgrid_obs)
            self.xgrid_obs_test = self.best_xgrid_obs
            print('best result')
        else:
            y_obs_test = self.RBFInteploation_test_s(self.xgrid_obs_test)
        test_loader = DataLoader(senseiver_loader_test(self.xgrid_obs_test[:, :1], self.xgrid_obs_test[:, 1:], y_obs_test,
                                                  self.y_test_exact_square, self.data_config),
                                 batch_size=None,
                                 pin_memory=True,
                                 shuffle=True,
                                 num_workers=1
                                 )
        self.net.eval()
        test_loss, test_num = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (inputs, coords, targets) in enumerate(test_loader):
                inputs, coords, targets = inputs.to(self.device), coords.to(self.device), targets.to(self.device)
                pre = self.net(inputs, coords)
                # loss = self.criterion(pre, targets)
                loss = self.criterion_l2(pre, targets) / self.y_mean_denom
                test_loss += loss.item() * inputs.shape[0]
                test_num += inputs.shape[0]
        test_loss = test_loss / test_num
        print('test loss', test_loss)
        return test_loss

    def plot_obs_update_senseiver(self, output, targets, xgrid_obs, i_plot, val_loss, test_loss, plot_save):
        i = 0
        u_val_plot = f_npy(targets[i:i + 1, :]).reshape(-1, 1)
        u_pred_plot = f_npy(output[i:i + 1, :]).reshape(-1, 1)
        u_error_plot = abs(u_pred_plot - u_val_plot)
        self.plot_obs(args, xgrid_obs, u_pred_plot, u_error_plot, u_val_plot, i_plot, val_loss, test_loss, plot_save)

if __name__ == "__main__":
    torch.cuda.set_device(1)
    setup_seed(0)
    args = parses()
    args.model, args.test = 'senseiver', 1
    args.xs, args.y_exact_flatten, args.y_exact_square = CylinderDataset(index=[i for i in range(5000)])
    args.y_exact_flatten, args.y_exact_square = args.y_exact_flatten / 11.0960, args.y_exact_square / 11.0960
    args.n_power, args.ns = args.y_exact_flatten.shape[0], args.y_exact_flatten.shape[1]
    args.num_obs = 8
    Sensor_i = Sensor_ini(args.y_exact_flatten.T, args.xs, args.num_obs)
    # args.xgrid_ini_methods = ['RandomSampler', 'UniformSampler', 'LHSampler', 'ConditionNumberSampler',
    #            'DeterminantBasedSampler', 'EnhancedClusteringSampler', 'CorrelationClusteringSampler']
    # args.xgrid_ini_methods = ['RandomSampler']
    args.xgrid_ini_methods = ['UniformSampler']
    for xgrid_ini_method in args.xgrid_ini_methods:
        args.xgrid_ini_method = xgrid_ini_method
        args.xgrid_obs_ini = Sensor_i.set_init(args)
        Sensor_i.plot_ini()
        NN_solver_rec = Solver_rec()
        NN_solver_rec.train(xobs_optimizer=True)
        NN_solver_rec.Predict(True)

