import torch
from sklearn.decomposition import PCA


class GappyPod():
    def __init__(self, data, map_size=(112, 192), n_components=40,
                 locations=[i * 150 for i in range(50)]):
        self.data = data.reshape(data.shape[0], -1)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.data.reshape(data.shape[0], -1))

        self.locations = locations
        self.map_size = map_size

        components = self.pca.components_
        means = self.pca.mean_

        self.component_mask = torch.from_numpy(components[:, locations]).float().cuda().unsqueeze(0)
        self.mean_mask = torch.from_numpy(means[locations]).float().cuda().reshape(-1, 1)
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def reconstruct(self, observations):
        component_mask = self.component_mask.repeat(observations.shape[0], 1, 1)
        observe = (observations.T - self.mean_mask).unsqueeze(dim=1).permute(2, 0, 1)
        coff_pre = torch.linalg.inv(component_mask @ component_mask.permute(0, 2, 1)) @ component_mask @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        recons = self.inverse_transform(coff_pre)
        # recons = recons.reshape(recons.shape[0], 2, self.map_size[0], self.map_size[2])
        return recons

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_


if __name__ == "__main__":
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    df = open('/home/ubuntu/zhaoxiaoyu/Data_total/cylinder/Cy_Taira.pickle', 'rb')
    data = np.transpose(pickle.load(df), (0, 3, 1, 2))[:4000, :, :, :]
    positions = [i * 43 for i in range(500)]
    gappy_pod = GappyPod(data=data, n_components=20, locations=positions)

    map = gappy_pod.reconstruct(torch.from_numpy(data[0, 0, :, :].reshape(-1)[positions].reshape(1, -1)).cuda().float())
    plt.imshow(map[0, 0, :, :].cpu().numpy(), cmap='jet')
    plt.colorbar()
    plt.savefig('cylinder.png')
