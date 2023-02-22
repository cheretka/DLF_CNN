import numpy as np
from keras.datasets import cifar10, cifar100, fashion_mnist, mnist
from keras.utils import to_categorical


def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_cifar100():
    selected_classes = ['cloud', 'forest', 'mountain', 'plain', 'sea', 'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 'bridge', 'castle', 'house',
                        'road', 'skyscraper', 'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train']

    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode = 'fine')

    selected_indices = np.isin(y_train, [selected_classes.index(c) for c in selected_classes]).reshape(-1)
    x_train = x_train[selected_indices]
    y_train = y_train[selected_indices]
    selected_indices = np.isin(y_test, [selected_classes.index(c) for c in selected_classes]).reshape(-1)
    x_test = x_test[selected_indices]
    y_test = y_test[selected_indices]

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255
    y_train = to_categorical(y_train, len(selected_classes))
    y_test = to_categorical(y_test, len(selected_classes))

    return x_train, y_train, x_test, y_test
