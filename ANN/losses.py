import numpy as np


class CrossEntropyLoss:
    def __call__( self, y_pred, y_true ):
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def gradient( self, y_pred, y_true ):
        gradient = (y_pred - y_true) / y_pred.size
        return gradient


class MSELoss:
    def __call__( self, y_pred, y_true ):
        return np.mean((y_pred - y_true) ** 2)

    def gradient( self, y_pred, y_true ):
        return 2 * (y_pred - y_true) / y_pred.size
