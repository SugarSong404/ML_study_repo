import numpy as np
class KMeans:
    def __init__(self, data=None,k_num = 2):
        if data is not None:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.data = self.__prepare_data(data)
            self.k_num = k_num
            self.num = self.data.shape[0]
            self.feature = self.data.shape[1]
            self.centroids = self.data[np.random.permutation(self.num)[:k_num],:]
        else:
            self.mean = np.array([])
            self.std = np.array([])

    def __prepare_data(self,data):
        return (data - self.mean) / self.std
    
    def train(self,it = 1000,loss_history = True):
        self.loss = []
        for _ in range(it):
            closest_centroid = self.__find_closest_centroid(self.data)
            new_centroids = self.__centroid_update(closest_centroid)
            if np.allclose(self.centroids, new_centroids, atol=1e-4):break
            if loss_history:self.loss.append(self.__calculate_loss())
            self.centroids = new_centroids

    def __centroid_update(self, closest_centroid):
        return np.array([
            self.data[closest_centroid == c].mean(axis=0) if np.any(closest_centroid == c) else np.zeros(self.feature)
            for c in range(self.k_num)
        ])

    def __find_closest_centroid(self, data):
        distances = np.linalg.norm(data[:, np.newaxis, :] - self.centroids, axis=2)
        closest_centroid = np.argmin(distances, axis=1)
        return closest_centroid

    def predict(self,data):
        new_data = self.__prepare_data(data)
        return self.__find_closest_centroid(new_data)
    
    def score(self):
        return self.loss[-1]
    
    def __calculate_loss(self):
        closest_centroid = self.__find_closest_centroid(self.data)
        loss = 0
        for p in range(self.data.shape[0]):
            centroid_idx = int(closest_centroid[p])
            loss += np.sum((self.data[p, :] - self.centroids[centroid_idx, :]) ** 2)
    
        return loss
    
    def silhouette_score(self):
        closest_centroid = self.__find_closest_centroid(self.data).astype(int).flatten()
    
        a = np.zeros(self.data.shape[0])
        for cluster in range(self.k_num):
            cluster_points = self.data[closest_centroid == cluster]
            if len(cluster_points) > 1:
                distances = np.linalg.norm(cluster_points[:, np.newaxis, :] - cluster_points, axis=2)
                np.fill_diagonal(distances, np.nan)
                a[closest_centroid == cluster] = np.nanmean(distances, axis=1)
    
        b = np.full(self.data.shape[0], np.inf)
        for cluster in range(self.k_num):
            cluster_points = self.data[closest_centroid == cluster]
            for other_cluster in range(self.k_num):
                if cluster != other_cluster:
                    other_cluster_points = self.data[closest_centroid == other_cluster]
                    if len(other_cluster_points) > 0:
                        distances = np.linalg.norm(cluster_points[:, np.newaxis, :] - other_cluster_points, axis=2)
                        b[closest_centroid == cluster] = np.minimum(
                            b[closest_centroid == cluster],
                            np.mean(distances, axis=1)
                        )
        silhouette_scores = np.where(
            a < b, 1 - a / b, 
            np.where(a == b, 0, b / a)
        )
    
        return np.mean(silhouette_scores)


    def save_model(self, filepath):
        np.savez(filepath, feature=self.feature, mean=self.mean, std=self.std, centroids = self.centroids, k_num = self.k_num)

    def load_model(self, filepath):
        loaded_data = np.load(filepath)
        self.mean = loaded_data['mean']
        self.std = loaded_data['std']
        self.centroids = loaded_data['centroids']
        self.k_num = loaded_data['k_num']
        self.feature = loaded_data['feature']

