# cython: language_level=3
import pyximport
pyximport.install()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import trange
import numpy as np
import copy
from torch.autograd import grad
import torch
import sys
sys.path.append('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO/cylinder2D')
from sensor_ini import Sensor_ini
from CylinderDataset import CylinderDataset
from DSPO_optimizer_cy import Solver_basic
sys.path.append('F:/pycharm_code/working_code/Differentiable_sensor_optimization/DSPSO')
from model.mlp import MLP
from utils.utils import setup_seed
from utils.default_options import parses

f_npy = lambda x: x.cpu().detach().numpy()
f_tensor = lambda x: torch.tensor(x).float().to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Solver_rec(Solver_basic):
    def __init__(self,):
        super().__init__(args)
        self.net = MLP(layers=[args.num_obs, 64, 64, 64, args.ns]).to(self.device)

    def train(self, xobs_optimizer):
        args.epochs, iter_obs_update = 300, 100
        # args.epochs, iter_obs_update = 300, 30
        # args.epochs, iter_obs_update = 3, 2
        xgrid_obs = self.xgrid_obs_ini
        if not xobs_optimizer:
            iter_obs_update = 1
        for xgrid_obs_epoch in range(iter_obs_update):
            y_obs_train = self.RBFInteploation_train(xgrid_obs)
            dydx1_val, dydx2_val, y_obs_val = self.RBFInteploation_val(xgrid_obs)
            dydx1_val_add = torch.where(dydx1_val > 0, 1e-2, -1e-2)
            dydx2_val_add = torch.where(dydx2_val > 0, 1e-2, -1e-2)
            dydx1_val += dydx1_val_add
            dydx2_val += dydx2_val_add

            train_loader, val_loader = (self.set_dataset(y_obs_train, self.y_train_exact),
                                        self.set_dataset(y_obs_val, self.y_val_exact, shuffle=False))
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
            loss_history = []
            # Build optimize
            pbar = trange(args.epochs, desc='Epochs')
            for epoch in pbar:
                # Training procedure
                train_loss, train_num = 0., 0.
                for i, (inputs, outputs) in enumerate(train_loader):
                    inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                    pre = self.net(inputs)
                    loss = self.criterion(outputs, pre)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * inputs.shape[0]
                    train_num += inputs.shape[0]
                train_loss = self.std*train_loss / train_num
                self.scheduler.step()
                # val_loss, val_num = 0., 0.
                # for i, (inputs, outputs) in enumerate(val_loader):
                #     inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                #     with torch.no_grad():
                #         pre = self.net(inputs)
                #     loss = self.criterion(outputs, pre)
                #     val_loss += loss.item() * inputs.shape[0]
                #     val_num += inputs.shape[0]
                # val_loss = val_loss / val_num
                # loss_history.append([train_loss, val_loss])
                # pbar.set_postfix({'Train Loss': train_loss, 'Val Loss': val_loss})
                loss_history.append([train_loss])
                pbar.set_postfix({'Train Loss': train_loss})

            self.xgrid_obs_test = copy.deepcopy(xgrid_obs)
            test_loss = self.Predict(loading=False)
            self.xgrid_obs_log.append([self.xgrid_obs_test])

            val_loss, val_num, grad_y_obs_val = 0., 0., []
            for i, (inputs, outputs) in enumerate(val_loader):
                inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                inputs.requires_grad_(True)
                pre = self.net(inputs)
                loss = self.criterion(outputs, pre)
                val_loss += loss.item() * inputs.shape[0]
                val_num += inputs.shape[0]
                grad_y_obs_batch = grad(loss, inputs, create_graph=False, retain_graph=True)[0]
                grad_y_obs_val.append(grad_y_obs_batch)
            grad_y_obs = torch.cat(grad_y_obs_val, dim=0) / len(val_loader)
            val_loss = self.std*val_loss / val_num
            # print(f'obs_updata {xgrid_obs_epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')

            # self.plot_obs_update(xgrid_obs, xgrid_obs_epoch, val_loss, test_loss, 'obs_log')
            # self.plot_train_loss(args, loss_history, val_loss, test_loss, xgrid_obs_epoch)

            if val_loss <= self.best_val:
                self.record_best(val_loss, test_loss, xgrid_obs, xgrid_obs_epoch)
                # self.plot_obs_update(xgrid_obs, xgrid_obs_epoch, val_loss, test_loss, 'best')
            else:
                self.worse_update += 1
                self.best_rec.append([self.best_val, self.best_test, val_loss])

            # print('worse_update', self.worse_update)
            with open(self.run_path, 'a') as f:
                f.write(f'\n\nobs_updata {xgrid_obs_epoch}: Train Loss: {train_loss}, Val Loss: {val_loss} \n')
                f.write(f'worse_update: {self.worse_update} \n')
                f.write(f'test loss: {test_loss}\n')

            grad_x1_grid_obs = (grad_y_obs * dydx1_val).sum(dim=0).reshape(-1, 1)
            grad_x2_grid_obs = (grad_y_obs * dydx2_val).sum(dim=0).reshape(-1, 1)
            grad_x_grid_obs = torch.cat([grad_x1_grid_obs, grad_x2_grid_obs], dim=1)
            self.dydx12_val = torch.cat([dydx1_val, dydx2_val], dim=1)
            xgrid_obs, self.exp_avg, self.exp_avg_sq, self.Step = self.optim_x_adam_revise(xgrid_obs, grad_x_grid_obs, self.exp_avg, self.exp_avg_sq,
                                                                     self.Step)
            if self.worse_update >= 6:
                self.net = self.best_net_para
                xgrid_obs = self.best_xgrid_obs
                self.worse_update = 0

        # self.plot_optimizer_loss(args, self.best_rec)
        self.save_error_log(args, self.best_net_para, self.best_xgrid_obs, self.best_xgrid_obs_log, self.best_rec)
        torch.save(self.xgrid_obs_log, self.args.fig_path + f'/model/{self.args.model}_obs_log.pth')

    def Predict(self, loading=True):
        if loading:
            self.net = torch.load(args.fig_path + f'/model/best_model.pth')
            self.best_xgrid_obs = torch.load(args.fig_path + f'/model/best_obs.pth')
            y_obs_test = self.RBFInteploation_test(self.best_xgrid_obs)
            print('best result')
        else:
            y_obs_test = self.RBFInteploation_test(self.xgrid_obs_test)
        test_loader = self.set_dataset(y_obs_test, self.y_test_exact, shuffle=False)
        self.net.eval()
        test_loss, test_num = 0.0, 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion_l2(outputs, targets)/self.y_mean_denom
                test_loss += loss.item() * inputs.size(0)
                test_num += inputs.shape[0]
        test_loss = test_loss / test_num
        print('test loss', test_loss)
        return test_loss

    def plot_obs_update(self, xgrid_obs, i_plot, val_loss, test_loss, plot_save):
        _, _, y_obs_val = self.RBFInteploation_val(xgrid_obs)
        i = 0
        u_val_plot = self.std*f_npy(self.y_val_exact[i:i + 1, :]).reshape(-1, 1)
        input_val = y_obs_val[i:i + 1, :]
        u_pred_plot = self.std*f_npy(self.net(input_val).reshape(-1, 1))
        u_error_plot = abs(u_pred_plot - u_val_plot)
        self.plot_obs(args, xgrid_obs, u_pred_plot, u_error_plot, u_val_plot, i_plot, val_loss, test_loss, plot_save)

if __name__ == '__main__':
    setup_seed(0)
    args = parses()
    args.model, args.test = 'mlp', 0
    args.xs, args.y_exact_flatten, _ = CylinderDataset(index=[i for i in range(5000)])
    args.n_power, args.ns = args.y_exact_flatten.shape[0], args.y_exact_flatten.shape[1]
    args.num_obs = 8
    Sensor_i = Sensor_ini(args.y_exact_flatten.T, args.xs, args.num_obs)
    # args.xgrid_ini_methods = ['ConditionNumberSampler',
    #            'DeterminantBasedSampler', 'EnhancedClusteringSampler', 'CorrelationClusteringSampler']
    args.xgrid_ini_methods = ['UniformSampler']
    for xgrid_ini_method in args.xgrid_ini_methods:
        args.xgrid_ini_method = xgrid_ini_method
        args.xgrid_obs_ini = Sensor_i.set_init(args)
        Sensor_i.plot_ini()
        NN_solver_rec = Solver_rec()
        # NN_solver_rec.train(xobs_optimizer = False)
        NN_solver_rec.Predict(True)
        print(1)
    # Test 2 RS 10
    # setup_seed(0)
    # args = parses()
    # args.model, args.test = 'mlp', 1
    # args.xs, args.y_exact_flatten, _ = CylinderDataset(index=[i for i in range(5000)])
    # args.n_power, args.ns = args.y_exact_flatten.shape[0], args.y_exact_flatten.shape[1]
    #
    # best_test_loss_log = []
    # for num_obs in [4, 8, 16]:
    #     args.num_obs = num_obs
    #     Sensor_i = Sensor_ini(args.y_exact_flatten.T, args.xs, args.num_obs)
    #     best_test_loss = []
    #     for i in range(30):
    #         args.ckpt = 'logs/RandomSampler1'
    #         args.test = i + 1
    #         args.xgrid_ini_method = 'RandomSampler'
    #         args.xgrid_obs_ini = Sensor_i.set_init_random(args)
    #         Sensor_i.plot_ini()
    #         NN_solver_rec = Solver_rec()
    #         NN_solver_rec.train(xobs_optimizer=False)
    #         test_loss = NN_solver_rec.Predict(True)
    #         best_test_loss.append(test_loss)
    #     best_test_loss_log.append(best_test_loss)
    #     print(args.num_obs, best_test_loss)
    # np.save('./logs/RandomSampler1/{}_best_test_loss.npy'.format(args.model), best_test_loss_log)