import tensorflow as tf
from tensorflow.keras import layers as KL


class CNN_encoder(tf.keras.Model):
    def __init__(self, num_layers, width, kernel_size, strides, act="relu"):
        """
        Initializes the Encoder model with the specified layer dimensions and activation function.
        """
        super(CNN_encoder, self).__init__()
        self.num_layers = num_layers
        self.width = width
        self.kernel_size = kernel_size
        self.strides = strides
        self.act = act

        # Creates a list of Dense layers for the encoder part, excluding the last dimension.
        self.CNN = [KL.Conv2D(self.width, kernel_size=self.kernel_size, strides=self.strides,
                              activation=self.act, name='encoder_%d' % i) for i in range(self.num_layers)]
        # The hidden layer representation
        self.hidden = KL.Dense(self.width, activation=self.act)

    def call(self, inputs):
        """
        Function to compute the output of the encoder given an input.
        Applies each layer in the encoder to the input in sequence.
        """
        h = inputs
        for i in range(self.num_layers):
            h = self.CNN[i](h)
        h = KL.Flatten()
        h = self.hidden(h)
        return h

