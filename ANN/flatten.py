import numpy as np
from ANN import layer


class Flatten(layer.Layer):

    def __init__( self ):
        self.input_shape = None

    def forward( self, X: np.ndarray ) -> np.ndarray:
        '''
        :param X: 4D shape (batch, height, width, channels)
        :return:  2D shape (batch, height * width * channels)
        '''
        self.input_shape = X.shape
        return X.reshape(self.input_shape[0], -1)

    def backward( self, gradient ):
        '''
        :param gradient:
        :return:  4D shape (batch, height, width, channels)
        '''
        return gradient.reshape(self.input_shape)
