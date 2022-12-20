import numpy as np


def identity_function(x):
    return x

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    
  
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def binary_cross_entropy_error(y, t):
    delta = 1e-7
    t = t.reshape(t.size,1)
    y = y.reshape(y.size,1)
 
    batch_size = y.shape[0]
    # -t*np.log(y+delta) +(1-t)*np.log(1-y +delta) / batch_size  #prevent -inf value from log funtion. so, we add the delta

    return -np.sum(t*np.log(y+delta) +(1-t)*np.log(1-y +delta)) / batch_size # np.sum returns scalar, and multipication is executed by elementwise
