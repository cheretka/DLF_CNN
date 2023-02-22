import numpy as np
from ANN import layer


class Dense(layer.Layer):

    def __init__( self, number_of_neurons, input_shape = None, activation = 'relu' ):
        self.output_shape = number_of_neurons
        self.input_shape = input_shape

        self.weights = None
        self.biases = None
        self.activation = activation
        self.learning_rate = 0.1

    def initialize( self ):
        '''
        weights: 2d shape (neurons_input, neurons_output)
                contains (-0.1, 0.1)
        biases: 1D shape (neurons_output)
                contains 0
        '''
        self.weights = np.random.uniform(low = -0.1,
                                         high = 0.1,
                                         size = (self.input_shape[-1], self.output_shape))
        self.biases = np.zeros(self.output_shape)


    def forward( self, X: np.ndarray ) -> np.ndarray:
        '''
        :param X: 2D shape (batch, neurons_input)
        :return:  2D shape (batch, neurons_output)
        '''
        # print("\nDense forward")

        if self.weights is None:
            if self.input_shape is None:
                self.input_shape = X.shape
            self.initialize()

        self.X = X
        # 2Dshape(batch, neurons_input) * 2Dshape(neurons_input, neurons_output) = 2Dshape(batch, neurons_output)
        self.output = np.dot(X, self.weights) + self.biases

        # aktywacja
        match self.activation:
            case 'relu':
                self.output_activ = np.maximum(self.output, 0)
            case 'sigmoid':
                self.output_activ = 1 / (1 + np.exp(-self.output))
            case 'tanh':
                self.output_activ = np.tanh(self.output)
            case 'softmax':
                # self.output_activ = np.exp(self.output) # give large values
                self.output_activ = np.exp(self.output - np.max(self.output, axis = 1, keepdims = True))
                self.output_activ /= np.sum(self.output_activ, axis = 1, keepdims = True)
            case _:
                self.output_activ = self.output

        return self.output_activ


    def backward( self, gradient: np.ndarray ) -> np.ndarray:
        '''
        :param gradient: 2D shape (batch, neurons_output)
        :return:         2D shape (batch, neurons_input)
        '''
        # print("Dense backward")

        # pochodna funckji aktywacji
        match self.activation:
            case 'relu':
                activ_der = np.where(self.output > 0, 1, 0)
            case 'sigmoid':
                activ_der = self.output_activ * (1 - self.output_activ)
            case 'tanh':
                activ_der = 1 - np.square(self.output_activ)
            case 'softmax':
                activ_der = 1  # activ_der = self.output_activ * ( 1 - self.output_activ)
            case _:
                activ_der = 1

        new_gradient = np.dot(gradient, self.weights.T)

        # aktualizacja wag
        delta = gradient * activ_der
        delta_weights = np.dot(self.X.T, delta)
        self.weights -= delta_weights * self.learning_rate
        delta_biases = np.sum(delta, axis = 0)
        self.biases -= delta_biases * self.learning_rate

        if np.amax(self.weights) > 1:
            print("Dense max weight value", np.amax(self.weights))

        return new_gradient
