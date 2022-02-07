"""Tests generator_mapping.py."""

import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from mlops.architecture.stylegan2.generator_mapping import GeneratorMapping


def test_no_conditioning_correct_input_shape() -> None:
    """Tests that GeneratorMapping has the correct input shape when no
    conditioning is used."""
    z_latent_size = 4
    #gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
    #model = Sequential([Input(shape=(z_latent_size,)), gen_mapping])
    #assert model.input_shape[1:] == (z_latent_size,)
    gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
    model = Sequential([gen_mapping(input_shape=(z_latent_size,))])
    assert model.input_shape[1:] == (z_latent_size,)


def test_cannot_instantiate_model_with_incorrect_input_shape() -> None:
    """Tests that a TensorFlow model cannot be instantiated with a
    GeneratorMapping that has an incorrect input shape."""
    z_latent_size = 4
    # TODO custom error.
    with pytest.raises(ValueError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
        _ = Sequential([Input(shape=(z_latent_size + 1,)), gen_mapping])
    with pytest.raises(ValueError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
        _ = Sequential([Input(shape=(z_latent_size - 1,)), gen_mapping])
    with pytest.raises(ValueError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
        _ = Sequential(gen_mapping(input_shape=(z_latent_size + 1,)))


def test_conditioning_correct_input_shape() -> None:
    """Tests that GeneratorMapping has the correct input shape when
    conditioning is used."""
    # TODO
    z_latent_size = 4
    gen_mapping = GeneratorMapping(z_latent_size=z_latent_size - 1,
                                   label_size=1,
                                   dtype='float64')
    arr = np.random.normal(size=(2, z_latent_size))
    print(gen_mapping(arr))
    print(gen_mapping(arr).shape)
    print(gen_mapping(arr).dtype)
    print(gen_mapping.mapping[1].dtype)
    from tensorflow.keras.models import Sequential
    model = Sequential(gen_mapping)
    print(model(arr))
    print(gen_mapping.input_shape)


def test_layer_outputs_specified_dtype() -> None:
    """Tests that the layer outputs tensors of the specified dtype."""
    z_latent_size = 4
    # Numpy data types.
    for dtype in [np.float16, np.float32, np.float64]:
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size,
                                       dtype=dtype)
        arr = np.random.normal(size=(2, z_latent_size))
        assert gen_mapping(arr).dtype == dtype
    # String data type.
    gen_mapping = GeneratorMapping(z_latent_size=z_latent_size,
                                   dtype='float64')
    arr = np.random.normal(size=(2, z_latent_size))
    assert gen_mapping(arr).dtype == np.float64


def test_sublayers_output_specified_dtype() -> None:
    """Tests that each encapsulated layer within GeneratorMapping has the
    specified dtype."""
    z_latent_size = 4
    # Layer dtypes are stored as strings; we could also pass numpy types to the
    # constructor, but comparison between the two would be more tedious.
    for dtype in ['float16', 'float32', 'float64']:
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size,
                                       dtype=dtype)
        # Layer properties are set lazily in build(), so we need to pass data
        # through once.
        arr = np.random.normal(size=(2, z_latent_size))
        _ = gen_mapping(arr)
        for layer in gen_mapping.mapping:
            assert layer.dtype == dtype

# TODO test trainable variables

# TODO test shapes

# TODO test conditioning labels

# TODO test normalization

# TODO test nonlinearity

# TODO test types
