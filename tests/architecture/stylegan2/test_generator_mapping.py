"""Tests generator_mapping.py."""

import numpy as np
from mlops.architecture.stylegan2.generator_mapping import GeneratorMapping


def test_init_creates_expected_attributes() -> None:
    """Tests that __init__ sets the expected object attributes."""
    # TODO
    assert False


def test_layer_outputs_specified_dtype() -> None:
    """Tests that the layer outputs tensors of the specified dtype."""
    # TODO


def test_no_conditioning_correct_input_shape() -> None:
    """Tests that GeneratorMapping has the correct input shape when no
    conditioning is used."""
    # TODO
    z_latent_size = 4
    gen_mapping = GeneratorMapping(z_latent_size=z_latent_size - 1, label_size=1)
    arr = np.random.normal(size=(2, z_latent_size))
    print(gen_mapping(arr))
    print(gen_mapping(arr).shape)

# TODO test trainable variables

# TODO test shapes

# TODO test conditioning labels

# TODO test normalization

# TODO test nonlinearity

# TODO test types
