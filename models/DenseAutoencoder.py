import tensorflow as tf
from tensorflow.keras import layers as KL


class DenseEncoder(tf.keras.Model):
    """
    Encoder part of an Autoencoder.

    This class defines an encoder model in a neural network, which is typically used
    in an autoencoder architecture for dimensionality reduction or feature extraction.

    Attributes:
        dims (list of int): List of integers where each integer represents the number of neurons
                            in each layer of the encoder.
        act (str): The activation function to use in each layer of the encoder.
        encoding (list of keras.layers.Dense): List of Dense layers that make up the encoder.

    Methods:
        call(inputs): Function to compute the output of the encoder given an input.
    """

    def __init__(self, dims=[63, 500, 500, 2000, 10], act="relu"):
        """
        Initializes the Encoder model with the specified layer dimensions and activation function.
        """
        super(DenseEncoder, self).__init__()
        self.dims = dims
        self.act = act
        # Creates a list of Dense layers for the encoder part, excluding the last dimension.
        self.encoding = [KL.Dense(self.dims[i + 1], activation=self.act, name='encoder_%d' % i) for i in
                         range(len(self.dims) - 2)]
        # The hidden layer representation
        self.hidden = KL.Dense(self.dims[-1], name='encoder_%d' % (len(self.dims) - 1))

    def call(self, inputs):
        """
        Function to compute the output of the encoder given an input.
        Applies each layer in the encoder to the input in sequence.
        """
        h = inputs
        for i in range(len(self.dims) - 2):
            h = self.encoding[i](h)
        h = self.hidden(h)
        return h
