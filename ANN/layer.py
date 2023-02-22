from abc import ABC, abstractmethod

import numpy as np

class Layer(ABC):

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, gradient):
        pass
