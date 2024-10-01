import math
import torch
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pyclustering.cluster import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
from tqdm import tqdm


class RandomSampler:
    def __init__(self, data, num):
        self.data = data
        self.num = num
        self.locations = None

    def sample(self):
        if np.ndim(self.data) == 2:
            self.locations = np.random.choice(self.data.shape[-1], self.num, replace=False)
        elif np.ndim(self.data) == 3:
            self.locations = np.random.choice(self.data.shape[-2] * self.data.shape[-1], self.num, replace=False)
        self.locations.sort()
        return self.locations.tolist()


class UniformSampler:
    def __init__(self, data, num, split=None):
        self.data = data
        self.num = num
        self.locations = None
        self.split = split

    def sample(self):
        if np.ndim(self.data) == 2:
            interval = math.floor(self.data.shape[-1] / (self.num + 1))
            self.locations = [interval * (i + 1) for i in range(self.num)]
        elif np.ndim(self.data) == 3:
            if self.split is None:
                interval = math.floor(self.data.shape[-2] * self.data.shape[-1] / (self.num + 1))
                self.locations = [interval * (i + 1) for i in range(self.num)]
            else:
                raw_interval = math.floor(self.data.shape[-2] / (self.split[0] + 1))
                column_interval = math.floor(self.data.shape[-1] / (self.split[1] + 1))
                self.locations = [self.data.shape[-1] * raw_interval * (i + 1) + column_interval * (j + 1) for i in
                                  range(self.split[0]) for j in range(self.split[1])]
        return self.locations


class LHSampler:
    def __init__(self, data, num):
        self.data = data
        self.num = num
        self.locations = None

    def sample(self):
        if np.ndim(self.data) == 2:
            sampler = qmc.LatinHypercube(d=1)
            locations = qmc.scale(sampler.random(self.num), l_bounds=[0], u_bounds=[self.data.shape[0] - 1])
            self.locations = [int(i) for i in locations]
        elif np.ndim(self.data) == 3:
            sampler = qmc.LatinHypercube(d=2)
            locations = qmc.scale(sampler.random(self.num), l_bounds=[0, 0],
                                  u_bounds=[self.data.shape[-2] - 1, self.data.shape[-1] - 1])
            self.locations = locations.astype(np.int32)
        self.locations = [self.locations[i, 0] * self.data.shape[-1] + self.locations[i, 1] for i in
                          range(self.locations.shape[0])]
        self.locations.sort()
        return self.locations


class ConditionNumberSampler:
    def __init__(self, data, num, n_components=25):
        self.data = data.reshape(data.shape[0], -1)
        self.num = num
        self.locations = None

        pca = PCA(n_components=n_components)
        self.pca = pca.fit(self.data)
        self.components = self.pca.components_
        self.mean = self.pca.mean_

    def sample(self, num_eva=500):
        locations = []
        std = np.std(self.data.T, axis=1)
        locations.append(np.argmax(std))
        for _ in tqdm(range(self.num - 1)):
            evaluations, start = [], 0
            while True:
                if start >= self.data.shape[1]:
                    break
                elif start + num_eva >= self.data.shape[1]:
                    candi_solution = np.array(locations).reshape(1, len(locations)).repeat(
                        self.data.shape[1] - start, 0)
                    append_location = np.array(list(range(start, self.data.shape[1]))).reshape(
                        self.data.shape[1] - start, 1)
                else:
                    candi_solution = np.array(locations).reshape(1, len(locations)).repeat(num_eva, 0)
                    append_location = np.array(list(range(start, start + num_eva))).reshape(num_eva, 1)

                if len(locations) == 0:
                    candi_solution = append_location
                else:
                    candi_solution = np.concatenate([candi_solution, append_location], axis=1)
                evaluations.append(self.com_condition_number(candi_solution).reshape(-1, 1))

                start += num_eva
            evaluations = np.concatenate(evaluations, axis=0).flatten()
            locations.append(np.argmin(evaluations))
        self.locations = locations
        self.locations.sort()
        return self.locations

    def com_condition_number(self, solutions):
        A = []
        for i in range(solutions.shape[0]):
            A.append(np.expand_dims(
                self.components[:, solutions[i]].T + self.mean[solutions[i]].reshape(-1, 1), axis=0
            ))
        A = np.concatenate(A, axis=0)
        condition_number = np.linalg.cond(A, p=2)
        return condition_number


class EnhancedClusteringSampler:
    def __init__(self, data, num, coor, n_clusters=500):
        self.data = data.reshape(data.shape[0], -1)
        self.num = num
        self.locations = None
        self.coor = coor
        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=9)
        self.voronoi = kmeans.fit_predict(coor)
        self.center = kmeans.cluster_centers_

    def sample(self):
        var = np.zeros(self.n_clusters)
        self.data_square = np.power(self.data, 2)
        for i in range(self.voronoi.shape[0]):
            var[self.voronoi[i]] += np.sum(self.data_square[:, i])
        var = var / self.data.shape[0]

        index = np.argpartition(var, -1 * self.num)[-1 * self.num:]

        positions = self.center[index, :]
        dis = np.linalg.norm(np.expand_dims(positions, axis=1) - np.expand_dims(self.coor, axis=0), axis=2)
        self.locations = np.argmin(dis, axis=1)
        self.locations.sort()
        return self.locations.tolist()


class ConditionNumberGPUSampler:
    def __init__(self, data, num, n_components=25):
        self.data = data.reshape(data.shape[0], -1)
        self.num = num
        self.locations = None

        pca = PCA(n_components=n_components)
        self.pca = pca.fit(self.data)
        self.components = self.pca.components_
        self.mean = self.pca.mean_

    def sample(self, num_eva=5000):
        locations = []
        std = np.std(self.data.T, axis=1)
        locations.append(np.argmax(std))
        for _ in tqdm(range(self.num - 1)):
            evaluations, start = [], 0
            while True:
                if start >= self.data.shape[1]:
                    break
                elif start + num_eva >= self.data.shape[1]:
                    candi_solution = np.array(locations).reshape(1, len(locations)).repeat(
                        self.data.shape[1] - start, 0)
                    append_location = np.array(list(range(start, self.data.shape[1]))).reshape(
                        self.data.shape[1] - start, 1)
                else:
                    candi_solution = np.array(locations).reshape(1, len(locations)).repeat(num_eva, 0)
                    append_location = np.array(list(range(start, start + num_eva))).reshape(num_eva, 1)

                if len(locations) == 0:
                    candi_solution = append_location
                else:
                    candi_solution = np.concatenate([candi_solution, append_location], axis=1)
                evaluations.append(self.com_condition_number(candi_solution).reshape(-1, 1))

                start += num_eva
            evaluations = np.concatenate(evaluations, axis=0).flatten()
            locations.append(np.argmin(evaluations))
        self.locations = locations
        self.locations.sort()
        return self.locations

    def com_condition_number(self, solutions):
        A = []
        for i in range(solutions.shape[0]):
            A.append(np.expand_dims(
                self.components[:, solutions[i]].T + self.mean[solutions[i]].reshape(-1, 1), axis=0
            ))
        A = np.concatenate(A, axis=0)
        condition_number = torch.linalg.cond(torch.from_numpy(A).cuda(), p=2)
        return condition_number.cpu().numpy()


class DeterminantBasedGPUSampler:
    def __init__(self, data, num, n_components=25):
        self.data = data.reshape(data.shape[0], -1)
        self.num = num
        self.locations = None

        pca = PCA(n_components=n_components)
        self.pca = pca.fit(self.data)
        self.components = self.pca.components_
        self.mean = self.pca.mean_

    def sample(self, num_eva=5000):
        locations = []
        for _ in tqdm(range(self.num)):
            positions, evaluations, start = [], [], 0
            while True:
                if start >= self.data.shape[1]:
                    break
                elif start + num_eva >= self.data.shape[1]:
                    selections = [s for s in range(start, self.data.shape[1]) if s not in locations]
                else:
                    selections = [s for s in range(start, start + num_eva) if s not in locations]
                candi_solution = np.array(locations).reshape(1, len(locations)).repeat(len(selections), 0)
                append_location = np.array(selections).reshape(len(selections), 1)

                if len(locations) == 0:
                    candi_solution = append_location
                else:
                    candi_solution = np.concatenate([candi_solution, append_location], axis=1)
                evaluations.append(self.com_det_criterion(candi_solution).reshape(-1, 1))

                positions = positions + selections
                start += num_eva
            evaluations = np.concatenate(evaluations, axis=0).flatten()
            locations.append(positions[np.argmax(evaluations)])
        self.locations = locations
        self.locations.sort()
        return self.locations

    def com_det_criterion(self, solutions):
        A = []
        for i in range(solutions.shape[0]):
            A.append(np.expand_dims(
                self.components[:, solutions[i]].T + self.mean[solutions[i]].reshape(-1, 1), axis=0
            ))
        A = np.concatenate(A, axis=0)
        if A.shape[1] < A.shape[2]:
            det_criterion = torch.linalg.det(torch.from_numpy(A @ np.transpose(A, (0, 2, 1))).cuda())
        else:
            det_criterion = torch.linalg.det(torch.from_numpy(np.transpose(A, (0, 2, 1)) @ A).cuda())
        return det_criterion.cpu().numpy()


def corr_distance(x1, x2):
    corr = np.corrcoef(x1, x2)
    return - 1 * corr[0, 1]


class CorrelationClusteringSampler:
    def __init__(self, data, num, coor, n_clusters=1000):
        self.data = data.reshape(data.shape[0], -1)
        self.num = num
        self.locations = None
        self.coor = coor

        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=9)
        self.voronoi = kmeans.fit_predict(coor)
        self.center = kmeans.cluster_centers_

    def sample(self):
        dis = np.linalg.norm(np.expand_dims(self.center, axis=1) - np.expand_dims(self.coor, axis=0), axis=2)
        center_locations = np.argmin(dis, axis=1)

        center_data = self.data[:, center_locations].T
        initial_centers = kmeans_plusplus_initializer(center_data, self.num).initialize()
        pc_km = kmeans.kmeans(center_data, initial_centers,
                              metric=distance_metric(type_metric.USER_DEFINED, func=corr_distance))
        pc_km.process()

        center = pc_km.get_centers()
        # dis = np.linalg.norm(np.expand_dims(center, axis=1) - np.expand_dims(center_data, axis=0), axis=2)
        dis = - 1 * np.corrcoef(center, center_data)
        self.locations = center_locations[np.argmin(dis[:self.num, -self.n_clusters:], axis=1)]
        self.locations.sort()
        return self.locations.tolist()


class DeterminantBasedSampler:
    def __init__(self, data, num, n_components=25):
        self.data = data.reshape(data.shape[0], -1)
        self.num = num
        self.locations = None

        pca = PCA(n_components=n_components)
        self.pca = pca.fit(self.data)
        self.components = self.pca.components_
        self.mean = self.pca.mean_

    def sample(self, num_eva=500):
        locations = []
        for _ in tqdm(range(self.num)):
            positions, evaluations, start = [], [], 0
            while True:
                if start >= self.data.shape[1]:
                    break
                elif start + num_eva >= self.data.shape[1]:
                    selections = [s for s in range(start, self.data.shape[1]) if s not in locations]
                else:
                    selections = [s for s in range(start, start + num_eva) if s not in locations]
                candi_solution = np.array(locations).reshape(1, len(locations)).repeat(len(selections), 0)
                append_location = np.array(selections).reshape(len(selections), 1)

                if len(locations) == 0:
                    candi_solution = append_location
                else:
                    candi_solution = np.concatenate([candi_solution, append_location], axis=1)
                evaluations.append(self.com_det_criterion(candi_solution).reshape(-1, 1))

                positions = positions + selections
                start += num_eva
            evaluations = np.concatenate(evaluations, axis=0).flatten()
            locations.append(positions[np.argmax(evaluations)])
        self.locations = locations
        self.locations.sort()
        return self.locations

    def com_det_criterion(self, solutions):
        A = []
        for i in range(solutions.shape[0]):
            A.append(np.expand_dims(
                self.components[:, solutions[i]].T + self.mean[solutions[i]].reshape(-1, 1), axis=0
            ))
        A = np.concatenate(A, axis=0)
        if A.shape[1] < A.shape[2]:
            det_criterion = np.linalg.det(A @ np.transpose(A, (0, 2, 1)))
        else:
            det_criterion = np.linalg.det(np.transpose(A, (0, 2, 1)) @ A)
        return det_criterion


class EfficientConditionNumberSampler:
    def __init__(self, data, num, coor, n_components=25, n_clusters=1000):
        self.data = data.reshape(data.shape[0], -1)
        self.num = num
        self.locations = None
        self.coor = coor

        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=9)
        self.voronoi = kmeans.fit_predict(coor)
        self.center = kmeans.cluster_centers_
        dis = np.linalg.norm(np.expand_dims(self.center, axis=1) - np.expand_dims(self.coor, axis=0), axis=2)
        self.center_locations = np.argmin(dis, axis=1)

        pca = PCA(n_components=n_components)
        self.pca = pca.fit(self.data)
        self.components = self.pca.components_
        self.mean = self.pca.mean_

    def sample(self, num_eva=200):
        locations = []
        std = np.std(self.data.T, axis=1)
        locations.append(np.argmax(std))
        for _ in tqdm(range(self.num - 1)):
            evaluations, start = [], 0
            while True:
                if start >= len(self.center_locations):
                    break
                elif start + num_eva >= len(self.center_locations):
                    candi_solution = np.array(locations).reshape(1, len(locations)).repeat(
                        len(self.center_locations) - start, 0)
                    append_location = self.center_locations[
                        np.array(list(range(start, len(self.center_locations))))].reshape(
                        len(self.center_locations) - start, 1)
                else:
                    candi_solution = np.array(locations).reshape(1, len(locations)).repeat(num_eva, 0)
                    append_location = self.center_locations[np.array(list(range(start, start + num_eva)))].reshape(
                        num_eva, 1)

                if len(locations) == 0:
                    candi_solution = append_location
                else:
                    candi_solution = np.concatenate([candi_solution, append_location], axis=1)
                evaluations.append(self.com_condition_number(candi_solution).reshape(-1, 1))

                start += num_eva
            evaluations = np.concatenate(evaluations, axis=0).flatten()
            locations.append(self.center_locations[np.argmin(evaluations)])
        self.locations = locations
        self.locations.sort()
        return self.locations

    def com_condition_number(self, solutions):
        A = []
        for i in range(solutions.shape[0]):
            A.append(np.expand_dims(
                self.components[:, solutions[i]].T + self.mean[solutions[i]].reshape(-1, 1), axis=0
            ))
        A = np.concatenate(A, axis=0)
        condition_number = np.linalg.cond(A, p=2)
        return condition_number


class EfficientDeterminantBasedSampler:
    def __init__(self, data, num, coor, n_components=25, n_clusters=1000):
        self.data = data.reshape(data.shape[0], -1)
        self.num = num
        self.locations = None
        self.coor = coor

        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=9)
        self.voronoi = kmeans.fit_predict(coor)
        self.center = kmeans.cluster_centers_
        dis = np.linalg.norm(np.expand_dims(self.center, axis=1) - np.expand_dims(self.coor, axis=0), axis=2)
        self.center_locations = np.argmin(dis, axis=1)

        pca = PCA(n_components=n_components)
        self.pca = pca.fit(self.data)
        self.components = self.pca.components_
        self.mean = self.pca.mean_

    def sample(self, num_eva=200):
        locations = []
        for _ in tqdm(range(self.num)):
            positions, evaluations, start = [], [], 0
            while True:
                if start >= len(self.center_locations):
                    break
                elif start + num_eva >= len(self.center_locations):
                    selections = [s for s in self.center_locations[start:].tolist() if s not in locations]
                else:
                    selections = [s for s in self.center_locations[start:start + num_eva].tolist() if
                                  s not in locations]
                candi_solution = np.array(locations).reshape(1, len(locations)).repeat(len(selections), 0)
                append_location = np.array(selections).reshape(len(selections), 1)

                if len(locations) == 0:
                    candi_solution = append_location
                else:
                    candi_solution = np.concatenate([candi_solution, append_location], axis=1)
                evaluations.append(self.com_det_criterion(candi_solution).reshape(-1, 1))

                positions = positions + selections
                start += num_eva
            evaluations = np.concatenate(evaluations, axis=0).flatten()
            locations.append(positions[np.argmax(evaluations)])
        self.locations = locations
        self.locations.sort()
        return self.locations

    def com_det_criterion(self, solutions):
        A = []
        for i in range(solutions.shape[0]):
            A.append(np.expand_dims(
                self.components[:, solutions[i]].T + self.mean[solutions[i]].reshape(-1, 1), axis=0
            ))
        A = np.concatenate(A, axis=0)
        if A.shape[1] < A.shape[2]:
            det_criterion = np.linalg.det(A @ np.transpose(A, (0, 2, 1)))
        else:
            det_criterion = np.linalg.det(np.transpose(A, (0, 2, 1)) @ A)
        return det_criterion



if __name__ == '__main__':
    import pickle
    import h5py
    df = open('/home/ubuntu/zhaoxiaoyu/Data_total/cylinder/Cy_Taira.pickle', 'rb')
    data_u = np.transpose(pickle.load(df), (0, 3, 1, 2))[:4250, :, :, :]
    # f = h5py.File('/mnt/jfs/zhaoxiaoyu/Data_total/heat/temperature.h5', 'r')
    # data_u = f['u'][:2000, :, :, :]

    x, y = np.linspace(0, 0.1 * data_u.shape[-1], data_u.shape[-1]), np.linspace(0.1 * data_u.shape[-2], 0,
                                                                                 data_u.shape[-2])
    x, y = np.meshgrid(x, y)
    coor = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

    # sampler = RandomSampler(np.squeeze(data_u, axis=1), num=16)
    # sampler = UniformSampler(np.squeeze(data_u, axis=1), num=16, split=(4, 4))
    # sampler = LHSampler(np.squeeze(data_u, axis=1), num=16)
    # sampler = EfficientConditionNumberSampler(data_u, coor=coor, num=8, n_components=50)
    # sampler = EfficientDeterminantBasedSampler(data_u, coor=coor, num=16, n_components=50)
    # sampler = EnhancedClusteringSampler(data_u, num=16, coor=coor, n_clusters=500)
    sampler = CorrelationClusteringSampler(data_u, num=4, coor=coor, n_clusters=500)
    locations = sampler.sample()

    # import scipy.io as sio
    # sio.savemat('Data_total.mat', {'Data_total': sampler.center})
    # Plotting
    print(locations)
    data = np.zeros((data_u.shape[-2], data_u.shape[-1])).flatten()
    data[locations] = 1
    plt.subplot(121)
    plt.imshow(data.reshape(data_u.shape[-2], data_u.shape[-1]))
    plt.subplot(122)
    plt.imshow(data_u[0, 0, :, :])
    plt.show()

    for i in range(min(9, len(locations))):
        plt.subplot(3, 3, i + 1)
        plt.plot(data_u.reshape(data_u.shape[0], -1)[:, locations[i]])
    plt.show()
