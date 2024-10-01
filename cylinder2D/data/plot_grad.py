# cython: language_level=3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from Differentiable_sensor_optimization.DSPSO.utils.default_options import parses
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pickle
import copy
from torch.autograd import grad
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import sys

f_npy = lambda x: x.cpu().detach().numpy()
f_tensor = lambda x: torch.tensor(x).float().to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    grad = np.load('cy_data_grad.npz')
    gradx = grad['dydx1']
    grady = grad['dydx2']
    xs1 = grad['xs_1']
    xs2 = grad['xs_2']

    for i in range(0, 5000, 100):
        plt.scatter(xs1, xs2, c=abs(gradx[:, i].reshape(112, 192)), cmap='rainbow', vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()
        out = gradx[:, i].reshape(112, 192)
        print()