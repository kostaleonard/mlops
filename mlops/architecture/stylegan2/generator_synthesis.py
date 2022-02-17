"""Contains the GeneratorSynthesis class."""

from typing import Any, Tuple, Optional
from tensorflow.keras.layers import Layer

NUM_CHANNELS_RGB = 3
LAST_LAYER_RESOLUTION = 1024
DEFAULT_FMAP_BASE = 16 << 10
DEFAULT_FMAP_DECAY = 1.0
DEFAULT_FMAP_MIN = 1
DEFAULT_FMAP_MAX = 512
DEFAULT_RESAMPLE_KERNEL = (1, 3, 3, 1)


class GeneratorSynthesis(Layer):
    """The synthesis sub-network that is used in the StyleGAN2 generator.
    Transforms the disentangled latent code w into images at various
    resolutions (4 x 4 to 1024 x 1024 in the StyleGAN2 paper)."""

    def __init__(
            self,
            name: str = 'generator_synthesis',
            dtype: Any = 'float32',
            num_channels: int = NUM_CHANNELS_RGB,
            resolution: int = LAST_LAYER_RESOLUTION,
            fmap_base: float = DEFAULT_FMAP_BASE,
            fmap_decay: float = DEFAULT_FMAP_DECAY,
            fmap_min: int = DEFAULT_FMAP_MIN,
            fmap_max: int = DEFAULT_FMAP_MAX,
            randomize_noise: bool = True,
            nonlinearity: str = 'lrelu',
            resample_kernel: Optional[Tuple[int, ...]] =
                DEFAULT_RESAMPLE_KERNEL,
            fused_modconv: bool = True,
            **kwargs: Any) -> None:
        """Instantiates the object.

        :param name: The name of the layer.
        :param dtype: The data type to use for activations and outputs.
        :param  num_channels: The number of output channels; default is 3 for
            RGB.
        :param resolution: The resolution of the last layer output, in pixels.
            The output will be an image of shape resolution x resolution x
            num_channels. Must be a power of 2.
        :param fmap_base: The number of feature maps in the first layer, not
            accounting for clipping to [fmap_min, fmap_max]. At each stage, the
            resolution of the image is doubled and the number of feature maps
            decreases (by default, halved).
        :param fmap_decay: The log2 feature map reduction when the resolution
            is doubled. If set to default of 1.0, the number of feature maps is
            halved.
        :param fmap_min: The minimum number of feature maps in any layer.
        :param fmap_max: The maximum number of feature maps in any layer.
        :param randomize_noise: If True, use random noise inputs; otherwise,
            read noise inputs from fixed noise variables.
        :param nonlinearity: The activation function for each layer: 'relu',
            'lrelu', etc.
        :param resample_kernel: TODO
        :param fused_modconv: TODO
        :param kwargs: Layer kwargs.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        # TODO we are inferring d_latent_size from the input to build
        # TODO resolution must be a power of 2
