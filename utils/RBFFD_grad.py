# cython: language_level=3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from math import pi
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import pyximport
pyximport.install()

import sys
sys.path.append('/mnt/jfs/liuxu/idrl_working_code/Differentiable_sensor_optimization/DSPSO/utils')
# from rbf.sputils import expand_rows
# from rbf.pde.fd import weight_matrix
from rbf.pde.fd import weight_matrix
# from rbf.pde.geometry import contains
# from rbf.pde.nodes import poisson_disc_nodes

class RBFFD_grad:
    def __init__(self, order=2, n=25, phi='phs3'):
        self.phi = phi
        self.order = order
        self.n = n
    def rbf_fd_grad_1d(self, x_nodes, x_nodes_total, u_nodes_total):
        if x_nodes_total.shape[0] != u_nodes_total.shape[0]:
            raise ValueError("wrong dimension")
        A_interior = weight_matrix(
            x=x_nodes,
            p=x_nodes_total,
            n=self.n,
            diffs=[1],
            phi=self.phi,
            order=self.order)
        A_interior = A_interior.toarray()

        return A_interior @ u_nodes_total

    def rbf_fd_1d(self, x_nodes, x_nodes_total, u_nodes_total):
        if x_nodes_total.shape[0] != u_nodes_total.shape[0]:
            raise ValueError("wrong dimension")
        A_interior = weight_matrix(
            x=x_nodes,
            p=x_nodes_total,
            n=self.n,
            diffs=[0],
            phi=self.phi,
            order=self.order)
        A_interior = A_interior.toarray()

        return A_interior @ u_nodes_total

    def rbf_fd_2d(self, x_nodes, x_nodes_total, u_nodes_total):
        if x_nodes_total.shape[0] != u_nodes_total.shape[0]: # u_nodes_total, ns x n_fun
            raise ValueError("wrong dimension")
        A_interior = weight_matrix(
            x=x_nodes,
            p=x_nodes_total,
            n=self.n,
            diffs=[0, 0],
            phi=self.phi,
            order=self.order)
        A_interior = A_interior.toarray()

        return A_interior @ u_nodes_total

    def rbf_fd_grad_2d(self, x_nodes, x_nodes_total, u_nodes_total):
        if x_nodes_total.shape[0] != u_nodes_total.shape[0]:
            raise ValueError("wrong dimension")
        A_interior_x = weight_matrix(
            x=x_nodes,
            p=x_nodes_total,
            n=self.n,
            diffs=[[1, 0], [0, 0]],
            phi=self.phi,
            order=self.order)
        A_interior_y = weight_matrix(
            x=x_nodes,
            p=x_nodes_total,
            n=self.n,
            diffs=[[0, 0], [0, 1]],
            phi=self.phi,
            order=self.order)

        A_interior_x = A_interior_x.toarray()
        A_interior_y = A_interior_y.toarray()
        A_interior_x = A_interior_x - np.eye(A_interior_x.shape[0],A_interior_x.shape[1], dtype=np.float32)
        A_interior_y = A_interior_y - np.eye(A_interior_x.shape[0],A_interior_x.shape[1], dtype=np.float32)
        return A_interior_x @ u_nodes_total, A_interior_y @ u_nodes_total

    def rbf_fd_grad2_2d(self, x_nodes, x_nodes_total, u_nodes_total):
        if x_nodes_total.shape[0] != u_nodes_total.shape[0]:
            raise ValueError("wrong dimension")
        A_interior_x = weight_matrix(
            x=x_nodes,
            p=x_nodes_total,
            n=self.n,
            diffs=[[2, 0], [0, 0]],
            phi=self.phi,
            order=self.order)
        A_interior_y = weight_matrix(
            x=x_nodes,
            p=x_nodes_total,
            n=self.n,
            diffs=[[0, 0], [0, 2]],
            phi=self.phi,
            order=self.order)

        A_interior_x = A_interior_x.toarray()
        A_interior_y = A_interior_y.toarray()
        A_interior_x = A_interior_x - np.eye(A_interior_x.shape[0],A_interior_x.shape[1])
        A_interior_y = A_interior_y - np.eye(A_interior_x.shape[0],A_interior_x.shape[1])
        return A_interior_x @ u_nodes_total, A_interior_y @ u_nodes_total

def RBFFD_test_1d():
    f_1d = lambda x: np.sin(pi * x) * np.exp(-0.6 * x)
    df_1d = lambda x: pi * np.cos(pi * x) * np.exp(-0.6 * x) - (0.6 * np.sin(pi * x) * np.exp(-0.6 * x))
    RBFFD = RBFFD_grad()
    x_nodes = 10 * np.random.rand(10, 1).reshape(-1, 1)
    x_nodes_total = 10 * np.random.rand(400, 1).reshape(-1, 1)
    u_exact = f_1d(x_nodes_total)
    dudx = RBFFD.rbf_fd_grad_1d(x_nodes, x_nodes_total, u_exact)
    plt.scatter(x_nodes, dudx, alpha=0.5, label='Pred')
    plt.scatter(x_nodes_total, df_1d(x_nodes_total), alpha=0.2, label='Exact')
    plt.legend()
    plt.show()


def RBFFD_test_1d_two_fun():
    f_1d = lambda x: np.concatenate([np.sin(pi * x) * np.exp(-0.6 * x), np.sin(x)], axis=1)
    df_1d = lambda x: np.concatenate(
        [pi * np.cos(pi * x) * np.exp(-0.6 * x) - (0.6 * np.sin(pi * x) * np.exp(-0.6 * x)), np.cos(x)], axis=1)
    RBFFD = RBFFD_grad()
    x_nodes = 10 * np.random.rand(10, 1).reshape(-1, 1)
    x_nodes_total = 10 * np.random.rand(400, 1).reshape(-1, 1)
    u_exact = f_1d(x_nodes_total)
    dudx = RBFFD.rbf_fd_grad_1d(x_nodes, x_nodes_total, u_exact)
    plt.scatter(x_nodes, dudx[:, 0], alpha=0.5, label='Pred')
    plt.scatter(x_nodes_total, df_1d(x_nodes_total)[:, 0], alpha=0.2, label='Exact')
    plt.legend()
    plt.show()

    plt.scatter(x_nodes, dudx[:, 1], alpha=0.5, label='Pred')
    plt.scatter(x_nodes_total, df_1d(x_nodes_total)[:, 1], alpha=0.2, label='Exact')
    plt.legend()
    plt.show()


def RBFFD_test_2d():
    # f_2d = lambda x: np.cos(2 * pi * x[:, :1]) * np.sin(pi * x[:, 1:])
    # df_2d_x = lambda x: -2 * pi * np.sin(2 * pi * x[:, :1]) * np.sin(pi * x[:, 1:])
    # df_2d_y = lambda x: pi * np.cos(2 * pi * x[:, :1]) * np.cos(pi * x[:, 1:])

    f_2d = lambda x: np.sin(pi * x[:, :1]) * x[:, 1:]**4
    df_2d_x = lambda x:  pi * np.cos(pi * x[:, :1]) * x[:, 1:]**4
    df_2d_y = lambda x: 4 * np.sin(pi * x[:, :1]) * x[:, 1:]**3

    RBFFD = RBFFD_grad(order=5, n=100, phi='phs3')
    xs = np.linspace(0, 1, 100)
    xs, ys = np.meshgrid(xs, xs)
    x_nodes = 1 * np.random.rand(20, 2)
    x_nodes_total = np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1)], axis=1)
    u_exact = f_2d(x_nodes_total)
    dudx, dudy = RBFFD.rbf_fd_grad_2d(x_nodes_total, x_nodes_total, u_exact)
    dudx_exact = df_2d_x(x_nodes_total)
    dudy_exact = df_2d_y(x_nodes_total)
    error_grad_x = sum(abs(dudx-dudx_exact)) / 20
    error_grad_y = sum(abs(dudx - dudx_exact)) / 20
    # plt.imshow((dudx+u_exact).reshape(100, 100), cmap='seismic')
    plt.imshow((dudx).reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(dudx_exact.reshape(100,100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(abs(dudx-dudx_exact).reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()

    plt.imshow(dudy.reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(dudy_exact.reshape(100,100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(abs(dudy-dudy_exact).reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()


def RBFFD_test_2d_two_fun():
    # f_2d = lambda x: np.cos(2 * pi * x[:, :1]) * np.sin(pi * x[:, 1:])
    # df_2d_x = lambda x: -2 * pi * np.sin(2 * pi * x[:, :1]) * np.sin(pi * x[:, 1:])
    # df_2d_y = lambda x: pi * np.cos(2 * pi * x[:, :1]) * np.cos(pi * x[:, 1:])

    f_2d = lambda x: np.concatenate([np.sin(pi * x[:, :1]) * x[:, 1:]**4, np.cos(2 * pi * x[:, :1]) * np.sin(pi * x[:, 1:])],axis=1)
    df_2d_x = lambda x:  np.concatenate([pi * np.cos(pi * x[:, :1]) * x[:, 1:]**4, -2 * pi * np.sin(2 * pi * x[:, :1]) * np.sin(pi * x[:, 1:])],axis=1)
    df_2d_y = lambda x: np.concatenate([4 * np.sin(pi * x[:, :1]) * x[:, 1:]**3, pi * np.cos(2 * pi * x[:, :1]) * np.cos(pi * x[:, 1:])],axis=1)

    RBFFD = RBFFD_grad(order=5, n=100, phi='phs3')
    xs = np.linspace(0, 1, 100)
    xs, ys = np.meshgrid(xs, xs)
    x_nodes = 1 * np.random.rand(20, 2)
    x_nodes_total = np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1)], axis=1)
    u_exact = f_2d(x_nodes_total)
    dudx, dudy = RBFFD.rbf_fd_grad_2d(x_nodes_total, x_nodes_total, u_exact)
    dudx_exact = df_2d_x(x_nodes_total)
    dudy_exact = df_2d_y(x_nodes_total)
    error_grad_x = sum(abs(dudx-dudx_exact)) / 20
    error_grad_y = sum(abs(dudx - dudx_exact)) / 20
    # plt.imshow((dudx+u_exact).reshape(100, 100), cmap='seismic')
    plt.imshow((dudx[:,:1]).reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(dudx_exact[:,:1].reshape(100,100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(abs(dudx[:,:1]-dudx_exact[:,:1]).reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()

    plt.imshow(dudy[:,1:].reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(dudy_exact[:,1:].reshape(100,100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(abs(dudy[:,1:]-dudy_exact[:,1:]).reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()

def RBFFD_test_grad2_2d():
    f_2d = lambda x: np.cos(2 * pi * x[:, :1]) * np.sin(pi * x[:, 1:])
    df_2d_xx = lambda x: -2 * pi *2 * pi* np.cos(2 * pi * x[:, :1]) * np.sin(pi * x[:, 1:])
    df_2d_y = lambda x: pi * np.cos(2 * pi * x[:, :1]) * np.cos(pi * x[:, 1:])

    RBFFD = RBFFD_grad(order=5, n=100, phi='phs3')
    xs = np.linspace(0, 1, 100)
    xs, ys = np.meshgrid(xs, xs)
    x_nodes = 1 * np.random.rand(20, 2)
    x_nodes_total = np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1)], axis=1)
    u_exact = f_2d(x_nodes_total)
    dudxx, dudy = RBFFD.rbf_fd_grad2_2d(x_nodes_total, x_nodes_total, u_exact)
    dudx_exact = df_2d_xx(x_nodes_total)

    plt.imshow((dudxx).reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow(dudx_exact.reshape(100,100), cmap='seismic')
    plt.colorbar()
    plt.show()
    plt.imshow((dudxx-dudx_exact).reshape(100, 100), cmap='seismic')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # RBFFD_test_1d_two_fun()
    RBFFD_test_2d()
    # RBFFD_test_2d_two_fun()
    # RBFFD_test_grad2_2d()