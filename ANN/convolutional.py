import numpy as np
from ANN import layer


class Conv2D(layer.Layer):

    def __init__( self, number_of_filters, kernel_size, strides = (1, 1), input_shape = None, activation = 'relu' ):
        self.number_of_filters = number_of_filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.strides = strides

        self.weights = None
        self.biases = None
        self.activation = activation
        self.learning_rate = 0.1

    def initialize( self ):
        '''
        weights: 4d shape (kernel_height, kernel_width, batch, number_of_filters)
                contains (-0.1, 0.1)
        biases: 1D shape (number_of_filters,)
                contains 0
        '''
        self.weights = np.random.uniform(low = -0.1,
                                         high = 0.1,
                                         size = (self.kernel_size[0], self.kernel_size[1], self.input_shape[-1], self.number_of_filters))
        self.biases = np.zeros(self.number_of_filters)


    def forward( self, X: np.ndarray ) -> np.ndarray:
        """
        :param X:
        :return:
        """
        # print("\nConv2D forward")
        self.X = np.array(X, copy = True)

        if self.weights is None:
            if self.input_shape is None:
                self.input_shape = X.shape
            self.initialize()

        input_size, input_height, input_width, input_channels = X.shape

        output_height = int(((input_height - self.kernel_size[0]) / self.strides[0]) + 1)
        output_width = int(((input_width - self.kernel_size[1]) / self.strides[1]) + 1)
        self.output = np.zeros((input_size, output_height, output_width, self.number_of_filters))

        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.strides[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.strides[1]
                w_end = w_start + self.kernel_size[1]
                self.output[:, h, w, :] = np.sum(X[:, h_start:h_end, w_start:w_end, :, np.newaxis]
                                                 * self.weights[np.newaxis, :, :, :],
                                                 axis = (1, 2, 3))

        self.output += self.biases

        # aktywacja
        match self.activation:
            case 'relu':
                self.output_activ = np.maximum(self.output, 0)
            case 'sigmoid':
                self.output_activ = 1 / (1 + np.exp(-self.output))
            case 'tanh':
                self.output_activ = np.tanh(self.output)
            case 'softmax':
                # self.output_activ = np.exp(self.output)
                self.output_activ = np.exp(self.output - np.max(self.output, axis = (1, 2), keepdims = True))
                self.output_activ /= np.sum(self.output_activ, axis = (1, 2), keepdims = True)
            case _:
                self.output_activ = self.output

        return self.output_activ


    def backward( self, gradient: np.ndarray ) -> np.ndarray:
        # print("Conv2D backward")

        # pochodna funckji aktywacji
        match self.activation:
            case 'relu':
                activ_der = np.where(self.output > 0, 1, 0)
            case 'sigmoid':
                activ_der = self.output_activ * (1 - self.output_activ)
            case 'tanh':
                activ_der = 1 - np.square(self.output_activ)
            case 'softmax':
                activ_der = 1
            case _:
                activ_der = 1

        _, gradient_height, gradient_width, _ = gradient.shape
        new_gradient = np.zeros_like(self.X)
        delta_weights = np.zeros_like(self.weights)
        delta = gradient * activ_der

        for h in range(gradient_height):
            for w in range(gradient_width):

                h_start = h * self.strides[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.strides[1]
                w_end = w_start + self.kernel_size[1]

                new_gradient[:, h_start:h_end, w_start:w_end, :] += np.sum(self.weights[np.newaxis, :, :, :, :]
                                                                           * gradient[:, h:h + 1, w:w + 1, np.newaxis, :],
                                                                           axis = 4)

                delta_weights += np.sum(self.X[:, h_start:h_end, w_start:w_end, :, np.newaxis]
                                        * delta[:, h:h + 1, w:w + 1, np.newaxis, :],
                                        axis = 0)

        # aktualizacja wag
        self.weights -= delta_weights * self.learning_rate
        delta_biases = np.sum(delta, axis = (0, 1, 2))
        self.biases -= delta_biases * self.learning_rate

        if np.amax(self.weights) > 1:
            print("Conv2D max weight value", np.amax(self.weights))

        return new_gradient
