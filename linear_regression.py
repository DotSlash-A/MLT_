import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LinReg(object):
    '''
    Linear regression model
    -----------------------
    y = X@w
    X: A feature matrix
    w: weight vector
    y: label vector
    '''
    def __init__(self):
        self.t0 = 200
        self.t1 = 100000

    def predict(self, X:np.ndarray) -> np.ndarray:
        '''

        :param X: Feature matrix for given inputs
        :return: y, the output label vector as predicted by the given model
        '''

        y = X @ self.w
        return y

    def loss(self, X:np.ndarray, y:np.ndarray) -> float:
        '''
        Calculating loss for a model based on known labels
        :param X: Feature matrix for given inputs
        :param y: Output label vector as predicted by the given model
        :return: Loss
        '''

        e = y - self.predict(X)
        return (1/2) * (np.transpose(e) @ e)

    def rmse(self, X:np.ndarray, y:np.ndarray) -> float:
        '''Calculates root mean squared error of prediction wrt actual label (evaluation)

        :param X: Feature matrix for given inputs
        :param y: Output label as predicted by the given model
        :return: Loss
        '''

        return np.sqrt((2/X.shape[0]) * self.loss(X, y))

    def fit(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
        '''Estimates parameters of the linear regression model with normal equatiopns.

        :param X: Feature matrix
        :param y: output label vector
        :return: Weight vector
        '''
        self.w = np.linalg.pinv(X) @ y
        return self.w

    def calculate_gradient(self, X:np.ndarray, y:np.ndarray) ->np.ndarray:
        '''Calculates gradients of loss function wrt to weight vector on training set

        :param X: Feature matrix
        :param y: label vector
        :return: A vector of gradients
        '''
        return np.transpose(X) @ (self.predict(X)-y)

    def update_weights(self, grad:np.ndarray, lr:float) ->np.ndarray:
        '''Updates the weights based on the gradient of loss function

        weight updates are carried out with the following formula:
            w_new := w_old - lr * grad

        :param grad: gradient of loss wrt w
        :param lr: learning rate
        :return: updated weight vector
        '''
        return (self.w = lr*grad)

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def gd(self, X:np.ndarray, y:np.ndarray, num_epochs:int, lr:float) -> np.ndarray:
        '''Estimates parameters of linear regression model thru gradient descent.

        :param X: feature matrix for training data
        :param y: label vector for training data
        :param num_epochs: number of training steps
        :param lr: learning rate
        :return: weight vector: final weight vector
        '''
        self.w = np.zeros((X.shape  ))



















#%%
