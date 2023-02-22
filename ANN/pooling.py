import numpy as np
from ANN import layer


class MaxPooling2D(layer.Layer):

    def __init__( self, pool_size ):
        self.pool_size = pool_size

    def forward( self, X: np.ndarray ) -> np.ndarray:
        '''
        :param X: 4D shape (batch, height, width, channels)
        :return:  4D shape (batch, height/2, width/2, channels)
        '''
        # print("\nMaxPooling2D forward")
        self.X = X
        input_size, input_height, input_width, input_channels = X.shape

        output_height = input_height // self.pool_size[0]
        output_width = input_width // self.pool_size[1]

        self.output = np.zeros((input_size, output_height, output_width, input_channels))

        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.pool_size[0]
                h_end = (h + 1) * self.pool_size[0]
                w_start = w * self.pool_size[1]
                w_end = (w + 1) * self.pool_size[1]
                # shape (batch, height, width, channels)
                # np.max -> shape (batch, channels)
                self.output[:, h, w, ] = np.max(X[:, h_start:h_end, w_start:w_end, :], axis = (1, 2))

        return self.output


    def backward( self, gradient: np.ndarray ) -> np.ndarray:
        """
        :param gradient:
        :return:
        """
        # print("MaxPooling2D backward")
        batch_size, gradient_height, gradient_width, gradient_channels = gradient.shape
        new_gradient = np.zeros_like(self.X)

        for h in range(gradient_height):
            for w in range(gradient_width):
                h_start = h * self.pool_size[0]
                h_end = (h + 1) * self.pool_size[0]
                w_start = w * self.pool_size[1]
                w_end = (w + 1) * self.pool_size[1]

                mask = (self.X[:, h_start:h_end, w_start:w_end, :] == np.max(self.X[:, h_start:h_end, w_start:w_end, :], axis = (1, 2), keepdims = True))
                new_gradient[:, h_start:h_end, w_start:w_end, :] += gradient[:, h:h + 1, w:w + 1, :] * mask

        return new_gradient
