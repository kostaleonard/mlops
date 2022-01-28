"""Contains the GeneratorMapping class."""

from typing import Optional
from tensorflow.keras.models import Model

DEFAULT_D_LATENT_SIZE = 512
DEFAULT_MAPPING_LAYERS = 8
DEFAULT_MAPPING_FMAPS = 512
DEFAULT_MAPPING_LRMUL = 0.01


# TODO could be Sequential?
class GeneratorMapping(Model):
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
            using conditioning; otherwise None.
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
        super().__init__()
