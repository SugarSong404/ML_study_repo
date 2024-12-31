import numpy as np
import random
class LogisticRegression:
    def __init__(self, data=None, label=None, method='sigmoid',poly_degree=0, sin_degree=0):
        self.poly_degree = poly_degree
        self.sin_degree = sin_degree

        if data is not None:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.data = self.__prepare_data(data)
            self.type = np.unique(label)
            self.num = self.data.shape[0]
            self.feature = self.data.shape[1]
            self.label = self.__prepare_label(label)
            self.theta = np.zeros((self.feature, self.type.shape[0]))
            self.method = method
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

    def __prepare_label(self, label):
        new_label = np.zeros((self.num, self.type.shape[0]))
        label = label.ravel() 
        for i, t in enumerate(self.type):
            new_label[:, i] = (label == t).astype(int)
        return new_label
    
    def train(self, alpha, it=1000,batch_size=None):
        self.loss = []
        for _ in range(it):
            for index in range(self.type.shape[0]):
                if batch_size is None:self.__full_gradient_descent(alpha,index)
                elif batch_size == 0:self.__random_gradient_descent(alpha,index)
                elif batch_size > 0:self.__mini_gradient_descent(alpha,index,batch_size)
            self.loss.append(self.__calculate_loss(self.data,self.label))

    def __activation(self, data ,theta):
        if self.method == "sigmoid":
            return 1 / (1 + np.exp(-(data @ theta)))
        else:
            exp_z = np.exp((data @ theta))
            exp_total = np.array(exp_z.shape)
            for i in range(self.type.shape[0]):
                exp_total = exp_total + np.exp(data @ self.theta[:, i])
            return exp_z / exp_total

    def __random_gradient_descent(self, alpha,index):
        idx = random.randint(0, self.num - 1)
        x = self.data[idx, :].reshape(1, -1)
        y = self.label[idx ,index]
        this_theta = self.theta[:, index]
        
        dev = y - self.__activation(x,this_theta)
        theta = alpha * (x.T * dev)
        self.theta[:, index] = this_theta + theta.flatten()

    def __mini_gradient_descent(self,alpha,index,batch_size):
        indices = np.random.choice(self.num, batch_size, replace=False)

        x_batch = self.data[indices, :]
        y_batch = self.label[indices,index]
        this_theta = self.theta[:, index]

        dev = y_batch - self.__activation(x_batch,this_theta)

        theta = (alpha / batch_size) * (x_batch.T @ dev)
        self.theta[:, index] = this_theta + theta

    def __full_gradient_descent(self,alpha,index):
        this_theta = self.theta[:, index]
        dev = self.label[:,index] - self.__activation(self.data,this_theta)
        theta = (alpha / self.num) * (self.data.T @ dev)
        self.theta[:, index] = this_theta + theta

    def score(self, data, label):
        predictions = np.argmax(self.predict(data), axis=1)
        accuracy = np.mean(predictions == label.ravel())
        return accuracy
    
    def predict(self,data):
        new_data = self.__prepare_data(data)
        probs = []
        for i in range(self.type.shape[0]):
            probs.append(self.__activation(new_data,self.theta[:,i]))
        return np.array(probs).T
    
    def __calculate_loss(self, data, label):
        probs = []
        for i in range(self.type.shape[0]):
            probs.append(self.__activation(data,self.theta[:,i]))
        predictions = np.array(probs).T
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(label * np.log(predictions), axis=1))
        return loss


    def save_model(self, filepath):
        np.savez(filepath, theta=self.theta, mean=self.mean, std=self.std,
                 poly_degree=self.poly_degree, sin_degree=self.sin_degree,
                 type=self.type,method=self.method)

    def load_model(self, filepath):
        loaded_data = np.load(filepath)
        self.theta = loaded_data['theta']
        self.mean = loaded_data['mean']
        self.std = loaded_data['std']
        self.type = loaded_data['type']
        self.method = loaded_data['method']
        self.poly_degree = loaded_data['poly_degree'].item()
        self.sin_degree = loaded_data['sin_degree'].item()
        self.feature = self.theta.shape[0]

