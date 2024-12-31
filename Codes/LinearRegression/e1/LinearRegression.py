import numpy as np
import random
class LinearRegression:
    def __init__(self, data=None, label=None, poly_degree=0, sin_degree=0):
        self.poly_degree = poly_degree
        self.sin_degree = sin_degree

        if data is not None:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.data = self.__prepare_data(data)
            self.label = label
            self.num = self.data.shape[0]
            self.feature = self.data.shape[1]
            self.theta = np.zeros((self.feature, 1))
        else:
            self.mean = np.array([])
            self.std = np.array([])

    def __prepare_data(self,data):
        new_data = (data - self.mean) / self.std
        data_pre = np.hstack((np.ones((data.shape[0], 1)), new_data))
        data_poly = self.__poly_data(self.poly_degree,data_pre)
        return self.__sin_data(self.sin_degree,data_poly)
    
    def __poly_data(self, poly_degree, data_pre):
        if poly_degree <= 1 : return data_pre

        poly_datas = data_pre
        for degree in range(2, poly_degree + 1):
            poly_datas = np.concatenate((poly_datas,data_pre[:, 1:] ** degree),axis=1) 
        return poly_datas

    def __sin_data(self, sin_degree, data_poly):
        if sin_degree == 0:
            return data_poly

        bias_term = data_poly[:, :1]
        data_need = data_poly[:, 1:]

        sin_datas = []
        for degree in range(1, sin_degree + 1):
            sin_datas.append(np.sin(degree * data_need))

        sin_datas_combined = np.hstack([bias_term, data_poly, *sin_datas])
        return sin_datas_combined


    def train(self,alpha,it = 1000,batch_size=None):
        self.loss = []
        for _ in range(it):
            if batch_size is None:self.__full_gradient_descent(alpha)
            elif batch_size == 0:self.__random_gradient_descent(alpha)
            else:self.__mini_gradient_descent(alpha,batch_size)

            self.loss.append(self.__calculate_loss(self.data, self.label))

    def __random_gradient_descent(self, alpha):
        idx = random.randint(0, self.num - 1)
        x = self.data[idx, :].reshape(1, -1)
        y = self.label[idx]
        dev = y - (x @ self.theta)
        theta = alpha * (x.T * dev)
        self.theta = self.theta + theta

    def __mini_gradient_descent(self,alpha,batch_size):
        indices = np.random.choice(self.num, batch_size, replace=False)

        x_batch = self.data[indices, :]
        y_batch = self.label[indices]

        dev = y_batch - (x_batch @ self.theta)

        theta = (alpha / batch_size) * (x_batch.T @ dev)
        self.theta = self.theta + theta

    def __full_gradient_descent(self,alpha):
        dev = self.label - (self.data @ self.theta)
        theta = (alpha / self.num) * (self.data.T @ dev)
        self.theta = self.theta + theta

    
    def score(self, data, label):
        new_data = self.__prepare_data(data)
        predictions = new_data @ self.theta
        ss_total = ((label - label.mean()) ** 2).sum()
        ss_residual = ((label - predictions) ** 2).sum()
        r2_score = 1 - (ss_residual / ss_total)
        return r2_score

    def predict(self,data):
        new_data = self.__prepare_data(data)
        return new_data @ self.theta
    
    def __calculate_loss(self,data,label):
        dev = (data @ self.theta) - label
        return 0.5*(1/(data.shape[0]))*((dev.T @ dev)[0][0])

    def save_model(self, filepath):
        np.savez(filepath, theta=self.theta, mean=self.mean, std=self.std,
                 poly_degree=self.poly_degree, sin_degree=self.sin_degree)

    def load_model(self, filepath):
        loaded_data = np.load(filepath)
        self.theta = loaded_data['theta']
        self.mean = loaded_data['mean']
        self.std = loaded_data['std']
        self.poly_degree = loaded_data['poly_degree'].item()
        self.sin_degree = loaded_data['sin_degree'].item()
        self.feature = self.theta.shape[0]

