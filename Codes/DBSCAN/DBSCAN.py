import numpy as np
class DBSCAN:
    def __init__(self, data=None):
        if data is not None:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.data = self.__prepare_data(data)
            self.num = self.data.shape[0]
            self.feature = self.data.shape[1]
        else:
            self.mean = np.array([])
            self.std = np.array([])

    def __prepare_data(self, data):
        return (data - self.mean) / self.std

    def __find_neighbors(self, index):
        point = self.data[index]
        distances = np.linalg.norm(self.data - point, axis=1)
        neighbors = np.where(distances <= self.radius)[0]
        return neighbors

    def __neighbors_recursion(self, index, visited):
        stack = [index]
        cluster = []
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = 1
            cluster.append(self.data[current])
            neighbors = self.__find_neighbors(current)
            if len(neighbors) >= self.density:
                stack.extend(n for n in neighbors if not visited[n])
        return cluster

    def train(self, radius, density):
        self.radius = radius
        self.density = density
        self.noise = np.empty((0, self.feature))
        self.clusters = []
        visited = np.zeros(self.num, dtype=bool)
        for i in range(self.num):
            if not visited[i]:
                cluster = self.__neighbors_recursion(i, visited)
                if len(cluster) >= self.density:
                    self.clusters.append(np.array(cluster))
                else:
                    self.noise = np.vstack((self.noise, self.data[i]))

    def find_k_distance(self, k=4):
        distances = []
        for i in range(self.num):
            point = self.data[i]
            dists = np.linalg.norm(self.data - point, axis=1)
            dists_sorted = np.sort(dists)[1:]
            distances.append(dists_sorted[k-1])
        return np.sort(distances)

    def save_model(self, filepath):
        np.savez(filepath, mean=self.mean, std=self.std, radius=self.radius, density=self.density, clusters=np.array(self.clusters, dtype=object), noise=self.noise,feature = self.feature)

    def load_model(self, filepath):
        loaded_data = np.load(filepath, allow_pickle=True)
        self.mean = loaded_data['mean']
        self.std = loaded_data['std']
        self.radius = loaded_data['radius']
        self.density = loaded_data['density']
        self.clusters = list(loaded_data['clusters'])
        self.noise = loaded_data['noise']
        self.feature = loaded_data['feature']
