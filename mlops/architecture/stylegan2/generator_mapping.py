"""Contains the GeneratorMapping class."""

from typing import Optional, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LeakyReLU, ReLU

DEFAULT_Z_LATENT_SIZE = 512
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
    during training.

    # TODO remove this note
    The size of the latent code z is inferred based on the
    first input to the layer (set lazily during build())."""

    def __init__(self,
                 name: str = 'generator_mapping',
                 dtype: Any = 'float32',
                 z_latent_size: int = DEFAULT_Z_LATENT_SIZE,
                 label_size: Optional[int] = None,
                 d_latent_size: int = DEFAULT_D_LATENT_SIZE,
                 d_latent_broadcast: Optional[int] = None,
                 mapping_layers: int = DEFAULT_MAPPING_LAYERS,
                 mapping_fmaps: int = DEFAULT_MAPPING_FMAPS,
                 mapping_lrmul: float = DEFAULT_MAPPING_LRMUL,
                 mapping_nonlinearity: str = 'lrelu',
                 normalize_latents: bool = True,
                 **kwargs: Any) -> None:
        """Instantiates the object.

        :param name: The name of the layer.
        :param dtype: The data type to use for activations and outputs.
        :param z_latent_size: The length of the input latent vector z.
        :param label_size: Optional. The length of the conditioning labels if
            using conditioning; otherwise None. Passing 0 is equivalent to
            passing None.
        :param d_latent_size: The length of the disentangled latent vector w.
        :param d_latent_broadcast: Optional. If provided, the output of the
            layer will be of shape batch_size x d_latent_broadcast x
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
        :param kwargs: Layer kwargs.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.z_latent_size = z_latent_size
        self.label_size = label_size
        self.d_latent_size = d_latent_size
        self.d_latent_broadcast = d_latent_broadcast
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_lrmul = mapping_lrmul
        self.mapping_nonlinearity = mapping_nonlinearity
        self.normalize_latents = normalize_latents
        self.conditioning_weights = None
        self.mapping = None

    def build(self, input_shape):
        # TODO types and docstring
        if len(input_shape) != 2:
            # TODO custom error
            raise ValueError('Too many dimensions for generator mapping input')
        if self.label_size is not None and \
                input_shape[-1] != self.z_latent_size + self.label_size:
            # TODO custom error
            raise ValueError(f'Expected input second dimension to have size '
                             f'{self.z_latent_size} + {self.label_size} = '
                             f'{self.z_latent_size + self.label_size}, but '
                             f'got {input_shape[-1]}')
        if self.label_size is None and input_shape[-1] != self.z_latent_size:
            # TODO custom error
            raise ValueError(f'Expected input second dimension to have size '
                             f'{self.z_latent_size}, but got '
                             f'{input_shape[-1]}')
        if self.label_size:
            self.conditioning_weights = self.add_weight(
                name='conditioning_weights',
                shape=(self.label_size, self.z_latent_size),
                initializer='random_normal')
        self.mapping = []
        for layer_idx in range(self.mapping_layers):
            if layer_idx == self.mapping_layers - 1:
                layer_size = self.d_latent_size
            else:
                layer_size = self.mapping_layers
            self.mapping.append(Dense(layer_size, dtype=self.dtype))
            self.mapping.append(self._get_activation_layer())

    def call(self, inputs, **kwargs):
        """TODO types and docstring"""
        x = inputs
        if self.conditioning_weights is not None:
            z, y = tf.split(
                x,
                num_or_size_splits=[self.z_latent_size, self.label_size],
                axis=1
            )
            conditions = tf.matmul(y, self.conditioning_weights)
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

    def _get_activation_layer(self) -> Layer:
        if self.mapping_nonlinearity == 'lrelu':
            return LeakyReLU(alpha=LEAKY_RELU_ALPHA, dtype=self.dtype)
        if self.mapping_nonlinearity == 'relu':
            return ReLU(dtype=self.dtype)
        # TODO custom error
        raise ValueError(f'Unknown activation function: '
                         f'{self.mapping_nonlinearity}')

    def losses(self):
        # TODO this is just a rough idea
        return self.mapping_lrmul * super().losses()
