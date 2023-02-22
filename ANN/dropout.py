import numpy as np
from ANN import layer


class Dropout(layer.Layer):

    def __init__( self, rate ):
        self.rate = 1 - rate

    def forward( self, X: np.ndarray ) -> np.ndarray:
        '''
        :param X: 2D shape (batch, neurons_input)
        :return:  2D shape (batch, neurons_input)
        '''
        self.mask = (np.random.rand(*X.shape) < self.rate).astype(float)  # mask contains only 0 and 1
        return X * self.mask / self.rate

    def backward( self, gradient: np.ndarray ) -> np.ndarray:
        '''
        :param gradient: 2D shape (batch, neurons_input)
        :return:         2D shape (batch, neurons_input)
        '''
        return gradient * self.mask / self.rate
