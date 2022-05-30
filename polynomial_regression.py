import numpy as np

class BaseOptimizers():
    def __init__(self, optimizer = 'GD'):
        self.optimizer = optimizer

    def loss(self, X, w, y):
        assert X.shape[-1] == w.shape[0], 'Incompatible shapes'
        y_hat = X @ w
        loss_ = np.sum(np.square(y_hat - y))
        return loss_

    def lr_schedule(self,t):
        t0, t1 = 200, 100000
        return t0 / (t + t1)

    def gradient(self, X, w, y):
        assert X.shape[-1] == w.shape[0], 'Incompatible shapes'
        return X.T @ ((X @ w) - y)

    def gradient_descent(self, X, y,
                         verbose, epochs, lr):
        w0 = np.random.normal(0, 1, size=(X.shape[1],1))
        self.all_weights = []
        for epoch in range(epochs):
            if verbose:
                print('The Current Loss is :', self.loss(X, w0, y))
            self.all_weights.append(w0)
            w0 = w0 - lr*(self.gradient(X, w0, y))
        return w0

    def mini_batch_gd(self, X, y,
                      verbose, epochs, batch_size):
        w0 = np.random.normal(0, 1, size=(X.shape[1],1))
        t = 0
        self.all_weights = []
        for epoch in range(epochs):
            random_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[random_indices]
            y_shuffled = y[random_indices]
            for i in range(0, X.shape[0], batch_size):
                t = t + 1
                X_temp = X_shuffled[i:i+batch_size]
                y_temp = y_shuffled[i:i+batch_size]
                lr = self.lr_schedule(t)
                self.all_weights.append(w0)
                w0 = w0 - lr*(self.gradient(X_temp, w0, y_temp))
            if verbose:
                print(f'Epoch {epoch} Loss is :', self.loss(X, w0, y))
        return w0

    def stochastic_gd(self, X,
                      y, verbose, epochs):
        w0 = np.random.normal(0, 1, size=(X.shape[1],1))
        t = 0
        self.all_weights = []
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                random_index = np.random.randint(X.shape[0])
                t = t + 1
                X_temp = X[random_index:random_index+1]
                y_temp = y[random_index:random_index+1]
                lr = self.lr_schedule(t)
                self.all_weights.append(w0)
                w0 = w0 - lr*(self.gradient(X_temp, w0, y_temp))
            if verbose:
                print(f'Epoch {epoch} Loss is :', self.loss(X, w0, y))
        return w0
#%%
#from gradient_optimizers import BaseOptimizers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class LinearRegression(BaseOptimizers):
    def __init__(self,
                 optimizer = 'GD',
                 random_seed = 42):
        super().__init__(optimizer)
        self.random_seed = random_seed

    def add_dummy_feature(self, X):
        matrix_dummy = np.hstack((np.ones((X.shape[0], 1),
                                          dtype = X.dtype),
                                  X))
        return matrix_dummy

    def preprocess(self, X):
        X = self.add_dummy_feature(X)
        return X

    def train(self,
              X_train,
              y_train,
              epochs = 200,
              batch_size = 100,
              learning_rate = 0.001,
              verbose = False):
        assert batch_size < X_train.shape[0], 'batch size must be smaller than the number of data points'
        X_train = self.preprocess(X_train)
        if self.optimizer == 'GD':
            self.optimized_weights = self.gradient_descent(X_train, y_train,
                                                           verbose, epochs,
                                                           learning_rate)
        elif self.optimizer == 'MBGD':
            self.optimized_weights = self.mini_batch_gd(X_train, y_train,
                                                        verbose, epochs=epochs,
                                                        batch_size = batch_size)
        else:
            self.optimized_weights = self.stochastic_gd(X_train, y_train,
                                                        verbose, epochs=epochs)
        self.weights = self.all_weights

    def predict(self, X):
        X = self.add_dummy_feature(X)
        assert X.shape[-1] == self.optimized_weights.shape[0], 'Incompatible Shapes'
        self.predictions = X @ self.optimized_weights
        return self.predictions


# X, y = make_regression(n_samples = 10000)
# y = y.reshape(-1,1)
# x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

# model_gd = LinearRegression(optimizer = 'SGD')
# model_gd.train(x_train, y_train, epochs = 10, verbose = True, learning_rate = 0.0001)
# error = model_gd.predict(x_test) - y_test
# sum_squared_error = np.sum(np.square(error))
# print(sum_squared_error)
#%%
import numpy as np
#from linear_regression import LinearRegression
import itertools
import functools

class PolynomialRegression():
    def __init__(self, degrees = 2,
                 optimizer = 'GD',
                 random_seed = 42):
        self.degrees = degrees
        self.optimizer = optimizer

    def polynomial_transform(self, X):
        if X.shape == 1:
            X = X.reshape(-1,1)
        X = X.T
        transformed_features = []
        for degree in range(1, self.degrees+1):
            for item in itertools.combinations_with_replacement(X, degree):
                transformed_features.append(
                    np.array(functools.reduce(lambda x, y: x * y, item))
                )
        return np.array(transformed_features).T

    def train(self, X_train,
              y_train,
              epochs = 200,
              batch_size = 100,
              learning_rate = 0.001,
              verbose = False):
        X_train = self.polynomial_transform(X_train)
        self.model = LinearRegression(optimizer = self.optimizer)
        self.model.train(X_train, y_train,
                         epochs = epochs,
                         batch_size = batch_size,
                         learning_rate = learning_rate,
                         verbose = verbose)
        self.optimized_weights = self.model.optimized_weights

    def predict(self, X):
        X = self.polynomial_transform(X)
        self.model.predict(X)
        return self.model.predictions

polynomial_transform(np.array([[1,2],[3,4]]), degree=3)





#%%
