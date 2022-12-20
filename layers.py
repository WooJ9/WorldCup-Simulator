import numpy as np
from functions import *
from copy import deepcopy

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0) 
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class elu:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = x
        self.out[self.out <= 0] = np.exp(self.out[self.out <= 0]) - 1
        return self.out
    
    def backward(self, dout):
        self.out[self.out > 0] = 1
        self.out[self.out <= 0] += 1 
        return self.out * dout

class Affine:
    def __init__(self, W, b):
        self.W = deepcopy(W)
        #deepcopy...
        self.b = deepcopy(b)
        
        self.x = None
        self.original_x_shape = None 

        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape 
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x): 

        out = sigmoid(x)  # 1/(1+np.exp(-x))
        self.out = out

        return out

    def backward(self, dout): 
        dx = dout * (1.0 - self.out) * self.out

        return dx

# class SigmoidWithLoss:
#     def __init__(self):
#         self.loss = None # loss function
#         self.y = None    # return sigmoid
#         self.t = None    # is it onehotencoding??
        
#     def forward(self, x, t):
#         self.t = t
#         self.y = sigmoid(x) # 1/(1+np.exp(-x))
#         self.loss = binary_cross_entropy_error(self.y, self.t)
        
#         return self.loss

#     def backward(self, dout=1):
#         self.t = self.t.reshape(-1,1) # size of t shape : (batch_size, ) --> change size to (batch_size, 1)
#         batch_size = self.t.shape[0]
#         #dx = ((1-self.t)/(1-self.y) - self.t/self.y) * self.y * (1-self.y) 
#         dx = (self.y - self.t) / batch_size * dout
        
#         return dx

class Dropout:

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class BatchNormalization:
    

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None 

        self.running_mean = running_mean
        self.running_var = running_var  
        
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


#--------------------------------------------------------------------------------

class Binary_cross_entropy:

    def __init__(self):
        self.loss = None
        self.y = None 
        self.t = None

    def forward(self, y, t):
        self.t = t
        self.y = y # input is 1/(1+np.exp(-x)) which is sigmoid function
        self.loss = binary_cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1): 
        self.t = self.t.reshape(-1,1) # size of t shape : (batch_size, ) --> change size to (batch_size, 1)
        batch_size = self.t.shape[0]
        dx = (((1-self.t)/(1-self.y) - self.t/self.y) / batch_size) * dout
        
        return dx