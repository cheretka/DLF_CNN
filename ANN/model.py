import numpy as np


class Sequential:

    def __init__( self ):
        self.layers = []
        self.loss = None

    def add( self, layer ):
        self.layers.append(layer)

    def compile( self, loss ):
        self.loss = loss

    def fit( self, X, y, batch_size = 16, epochs = 10 ):

        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = 0, 0

            for b in range(0, len(X), batch_size):
                batch_X = X[b:b + batch_size]
                batch_y = y[b:b + batch_size]

                # Forward pass
                y_pred = self.predict(batch_X)
                # print("predicted ", y_pred[0])

                # Loss + Accuracy
                epoch_loss += self.loss(y_pred, batch_y)
                epoch_accuracy += self.get_accuracy(y_pred, batch_y)

                # Backward pass
                gradient = self.loss.gradient(y_pred, batch_y)
                self.backpropagate(gradient)

            print(f"Epoch {(epoch + 1)}, Loss: {(epoch_loss):.8f}, accuracy: {(epoch_accuracy):.1f} ({100 * (epoch_accuracy / len(X)):.1f}%)")

    def predict( self, X ):
        # print("X", X.shape)
        for layer in self.layers:
            X = layer.forward(X)
            # print("X", layer, X.shape, "min" , X.min(), "max", X.max())

        return X

    def backpropagate( self, gradient ):
        # print("gradient", gradient.shape)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            if np.all(gradient == 0):
                print("gradient 0 ", end = '')
            # print("\ngradient", np.all(gradient == 0), "\n", layer, gradient.shape)

    def get_accuracy( self, y_pred, y_true ):
        return np.sum(np.argmax(y_pred, axis = 1) == np.argmax(y_true, axis = 1))

    def evaluate( self, X, y ):
        y_pred = self.predict(X)
        loss = self.loss(y_pred, y)
        accuracy = self.get_accuracy(y_pred, y)
        return loss, accuracy
