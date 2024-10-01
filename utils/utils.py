import torch
import numpy as np
import random
import os

def setup_seed(seed):
    # torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.backends.cudnn.deterministic = True


def cre(truth, pre, ord):
    """
    Calculated relative error
    :param truth: (N,)
    :param pre: (N,)
    :param ord: 1, 2
    :return: (N,)
    """
    return torch.linalg.norm((truth - pre).flatten(1), ord, dim=1) / torch.linalg.norm(truth.flatten(1), ord, dim=1)


def generate_locations(data, observe_num=2, interval=2):
    """
    根据数据每个位置方差，生成测点位置。
    :param data: 物理场数据，(N, h, w)
    :param observe_num: 测点数量
    :param interval: 测点之间上下左右最小间隔
    :return: 测点位置，包含observe_num个测点位置的list
    """
    w, h = data.shape[2], data.shape[1]

    # 按照方差大小排序
    data = np.std(data, axis=0)
    argsort_index = np.flipud(np.argsort(data.flatten()))

    raw, col = np.linspace(0, h - 1, h), np.linspace(0, w - 1, w)
    col, raw = np.meshgrid(col, raw)
    col, raw = col.astype(np.int).flatten(), raw.astype(np.int).flatten()

    locations = []
    locations.append([raw[argsort_index[0]], col[argsort_index[0]]])
    for i in range(1, len(argsort_index)):
        if len(locations) < observe_num:
            cur_raw, cur_col = raw[argsort_index[i]], col[argsort_index[i]]
            flag = -1
            for [for_raw, for_col] in locations:
                if abs(for_raw - cur_raw) <= interval and abs(for_col - cur_col) <= interval:
                    flag = 1
            if flag == -1:
                locations.append([raw[argsort_index[i]], col[argsort_index[i]]])
        else:
            break
    return locations


def generate_locations_random(data, observe_num=2, interval=2):
    """
    根据数据每个位置方差，生成测点位置。
    :param data: 物理场数据，(N, h, w)
    :param observe_num: 测点数量
    :param interval: 测点之间上下左右最小间隔
    :return: 测点位置，包含observe_num个测点位置的list
    """
    w, h = data.shape[2], data.shape[1]

    # 按照方差大小排序
    data = np.std(data, axis=0)
    argsort_index = np.flipud(np.argsort(data.flatten()))

    raw, col = np.linspace(0, h - 1, h), np.linspace(0, w - 1, w)
    col, raw = np.meshgrid(col, raw)
    col, raw = col.astype(np.int).flatten(), raw.astype(np.int).flatten()

    locations = []
    locations.append([raw[argsort_index[0]], col[argsort_index[0]]])
    for i in range(1, len(argsort_index)):
        if len(locations) < observe_num:
            cur_raw, cur_col = raw[argsort_index[i]], col[argsort_index[i]]
            flag = -1
            for [for_raw, for_col] in locations:
                if abs(for_raw - cur_raw) <= interval or abs(for_col - cur_col) <= interval:
                    flag = 1
            if flag == -1:
                locations.append([raw[argsort_index[i]], col[argsort_index[i]]])
        else:
            break
    return locations

def flowmeter_get_noise_data(noise_scale, N_fun_test, num_obs, noise_type="none"):
    np.random.seed(0)
    if noise_type == "none":
        return 0
    elif noise_type == "normal":
        eps = np.random.normal(scale=noise_scale, size=(N_fun_test, num_obs))
        return eps
    elif noise_type == "contamined":
        eps1 = np.random.normal(scale=noise_scale, size=(N_fun_test, num_obs))
        eps2 = np.random.normal(scale=10*noise_scale, size=(N_fun_test, num_obs))
        return 0.8*eps1+0.2*eps2
    elif noise_type == "cauchy":
        eps = np.random.standard_cauchy(size=(N_fun_test, num_obs)) * noise_scale
        return eps
    else:
        raise NotImplementedError(f'noise type {noise_type} not implemented.')

if __name__ == '__main__':
    # eps = flowmeter_get_noise_data(0.01, 1000, 10, noise_type="cauchy")
    eps = np.random.poisson(0.1, 1000)
    print(1)
