import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from layers import *
from functions import *

class MultiLayerNet:
    def __init__(self,input_size,hidden_size_list,output_size,
                activation='elu',weight_init_std='he',use_dropout=False,
                dropout_ration=0.2, use_batchnorm=True):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.params = {}
        
        # initialize the weight
        self.__init_weight(weight_init_std)

        # generating layer
        activation_layer = {'sigmoid': Sigmoid, 'elu' : elu}
        self.layers = OrderedDict() 
        for i in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(i)] = Affine(self.params['W'+str(i)],
                                                    self.params['b'+str(i)])                                                 
            if self.use_batchnorm: # batch normalization - setting parameters and stage batch normalization layers
                self.params['gamma'+str(i)] = np.ones(hidden_size_list[i-1])
                self.params['beta'+str(i)] = np.zeros(hidden_size_list[i-1])
                self.layers['BatchNorm'+str(i)] = BatchNormalization(self.params['gamma'+str(i)],self.params['beta'+str(i)])
            
            self.layers['Activation_function'+str(i)] = activation_layer[activation]()
        
            if self.use_dropout: #dropout - prevent overfitting
                self.layers['Dropout'+str(i)] = Dropout(dropout_ration)

        # change the dimension to 1
        index = self.hidden_layer_num + 1
        self.layers['Affine'+str(index)] = Affine(self.params['W'+str(index)],self.params['b'+str(index)])
        self.layers['Activation_function'+str(index)] = activation_layer['sigmoid']()

        self.last_layer = Binary_cross_entropy() 


    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for i in range(1,len(all_size_list)):
            scale  = weight_init_std
            if str(weight_init_std).lower() in ('relu','he'):
                scale = np.sqrt(2.0 / all_size_list[i-1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[i-1])
            
            # initialize the value of W, b, and the size of input data by each layer
            # generate the random matrix and multiply the standard deviation  
            # multiplying scale because np.random.rand depends on standard normal distribution. 
            # (so we multiply sqrt(1/n) to make the standard deviation to sqrt(1/n))
            self.params['W' + str(i)] = scale * np.random.randn(all_size_list[i-1], all_size_list[i])
            self.params['b' + str(i)] = np.zeros(all_size_list[i])


    # train_flg is parameter which determine execution of dropout and batch normalization
    def predict(self, x, train_flg=False):

        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x
    

    def loss(self, x, t, train_flg=False):
        # x: input data
        # t: label

        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)
    
    # size of the parameters ->  x:(11032,34), y:(11032,1), t:(11032,1)
    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.around(y) # round np array
        t = t.reshape(-1,1)

        accuracy = np.sum(y==t) / float(x.shape[0])

        return accuracy

    def gradient(self, x, t):

        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # save the results
        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            grads['W'+str(i)] = self.layers['Affine'+str(i)].dW
            grads['b'+str(i)] = self.layers['Affine'+str(i)].db

            if self.use_batchnorm and i != self.hidden_layer_num + 1:
                grads['gamma'+str(i)] = self.layers['BatchNorm'+str(i)].dgamma
                grads['beta'+str(i)] = self.layers['BatchNorm'+str(i)].dbeta

        return grads