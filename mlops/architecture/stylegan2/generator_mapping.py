"""Contains the GeneratorMapping class."""

from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LeakyReLU, ReLU

DEFAULT_D_LATENT_SIZE = 512
DEFAULT_MAPPING_LAYERS = 8
DEFAULT_MAPPING_FMAPS = 512
DEFAULT_MAPPING_LRMUL = 0.01
LEAKY_RELU_ALPHA = 0.2
EPSILON = 1e-8


class GeneratorMapping(Layer):
    """The mapping sub-network used in the StyleGAN generator. Transforms the
    input latent code z into a disentangled latent code w. The input latent
    code z is drawn at random, e.g., from a multivariate Gaussian distribution.
    The disentangled latent code w contains style information that is learned
    during training."""

    def __init__(self,
                 z_latent_size: int,
                 label_size: Optional[int] = None,
                 d_latent_size: int = DEFAULT_D_LATENT_SIZE,
                 d_latent_broadcast: Optional[int] = None,
                 mapping_layers: int = DEFAULT_MAPPING_LAYERS,
                 mapping_fmaps: int = DEFAULT_MAPPING_FMAPS,
                 mapping_lrmul: float = DEFAULT_MAPPING_LRMUL,
                 mapping_nonlinearity: str = 'lrelu',
                 normalize_latents: bool = True,
                 dtype: str = 'float32') -> None:
        """Instantiates the object.

        :param z_latent_size: The length of the input latent vector z.
        :param label_size: Optional. The length of the conditioning labels if
            using conditioning; otherwise None. Passing 0 is equivalent to
            passing None.
        :param d_latent_size: The length of the disentangled latent vector w.
        :param d_latent_broadcast: Optional. If provided, the output of the
            model will be of shape batch_size x d_latent_broadcast x
            d_latent_size; otherwise, the output will be of shape batch_size x
            d_latent_size.
        :param mapping_layers: The number of fully connected layers through
            which to pass the latent code.
        :param mapping_fmaps: The number of neurons in each fully connected
            layer.
        :param mapping_lrmul: The learning rate multiplier for mapping layers.
        :param mapping_nonlinearity: The activation function for each layer:
            'relu', 'lrelu', etc.
        :param normalize_latents: Whether to normalize latent vectors before
            passing them through the mapping layers.
        :param dtype: The data type to use for activations and outputs.
        """
        # TODO mapping_lrmul
        super().__init__(dtype=dtype)
        if label_size:
            self.conditioning_weights = tf.Variable(
                initial_value=tf.random_normal_initializer(
                    shape=(label_size, z_latent_size)),
                trainable=True
            )
        else:
            self.conditioning_weights = None
        self.normalize_latents = normalize_latents
        self.d_latent_broadcast = d_latent_broadcast
        self.mapping = []
        input_shape = (z_latent_size,)
        for layer_idx in range(mapping_layers):
            layer_size = d_latent_size if layer_idx == mapping_layers - 1 \
                else mapping_fmaps
            self.mapping.append(
                Dense(layer_size,
                      dtype=dtype,
                      input_shape=input_shape)
            )
            self.mapping.append(
                GeneratorMapping._get_activation_function(mapping_nonlinearity)
            )
            input_shape = (layer_size,)

    def call(self, inputs, **kwargs):
        """TODO types and docstring"""
        x = inputs
        if self.conditioning_weights:
            z, y = x
            conditions = tf.matmul(y,
                                   self.conditioning_weights,
                                   dtype=self.dtype)
            x = tf.concat([z, conditions], axis=1)
        if self.normalize_latents:
            # Add epsilon for numerical stability in reciprocal sqrt.
            x *= tf.math.rsqrt(
                tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + EPSILON)
        for layer in self.mapping:
            x = layer(x)
        if self.d_latent_broadcast:
            x = tf.tile(x[:, np.newaxis], [1, self.d_latent_broadcast, 1])
        return x

    @staticmethod
    def _get_activation_function(mapping_nonlinearity: str) -> Layer:
        if mapping_nonlinearity == 'lrelu':
            return LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        if mapping_nonlinearity == 'relu':
            return ReLU()
        # TODO custom error
        raise ValueError(f'Unknown activation function: '
                         f'{mapping_nonlinearity}')
