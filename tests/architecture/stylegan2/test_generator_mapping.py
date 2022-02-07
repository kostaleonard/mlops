"""Tests generator_mapping.py."""

import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from mlops.architecture.stylegan2.generator_mapping import GeneratorMapping


def test_no_conditioning_correct_input_shape() -> None:
    """Tests that GeneratorMapping has the correct input shape when no
    conditioning is used."""
    z_latent_size = 4
    model = Sequential([
        Input(shape=(z_latent_size,)),
        GeneratorMapping(z_latent_size=z_latent_size)])
    assert model.input_shape[1:] == (z_latent_size,)
    model = Sequential([
        GeneratorMapping(z_latent_size=z_latent_size,
                         input_shape=(z_latent_size,))])
    assert model.input_shape[1:] == (z_latent_size,)


def test_conditioning_correct_input_shape() -> None:
    """Tests that GeneratorMapping has the correct input shape when
    conditioning is used."""
    z_latent_size = 3
    label_size = 1
    model = Sequential([
        Input(shape=(z_latent_size + label_size,)),
        GeneratorMapping(z_latent_size=z_latent_size,
                         label_size=label_size)])
    assert model.input_shape[1:] == (z_latent_size + label_size,)
    model = Sequential([
        GeneratorMapping(z_latent_size=z_latent_size,
                         label_size=label_size,
                         input_shape=(z_latent_size + label_size,))])
    assert model.input_shape[1:] == (z_latent_size + label_size,)


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


def test_correct_output_shape() -> None:
    """Tests that the layer has the correct output shape."""
    z_latent_size = 4
    d_latent_size = 8
    model = Sequential([
        GeneratorMapping(z_latent_size=z_latent_size,
                         d_latent_size=d_latent_size,
                         input_shape=(z_latent_size,))])
    assert model.output_shape[1:] == (d_latent_size,)


def test_broadcast_correct_output_shape() -> None:
    """Tests that the layer has the correct output shape when using
    d_latent_broadcast."""
    z_latent_size = 4
    d_latent_size = 8
    d_latent_broadcast = 3
    model = Sequential([
        GeneratorMapping(z_latent_size=z_latent_size,
                         d_latent_size=d_latent_size,
                         d_latent_broadcast=d_latent_broadcast,
                         input_shape=(z_latent_size,))])
    assert model.output_shape[1:] == (d_latent_broadcast, d_latent_size)


def test_layer_outputs_correct_dtype() -> None:
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


def test_sublayers_output_correct_dtype() -> None:
    """Tests that each encapsulated layer within GeneratorMapping has the
    specified dtype."""
    z_latent_size = 4
    # Layer dtypes are stored as strings; we could also pass numpy types to the
    # constructor, but comparison between the two would be more tedious.
    for dtype in ['float16', 'float32', 'float64']:
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size,
                                       dtype=dtype)
        # Layer properties are set lazily in build(), so we need to pass data
        # through once or call the method.
        arr = np.random.normal(size=(2, z_latent_size))
        _ = gen_mapping(arr)
        for layer in gen_mapping.mapping:
            assert layer.dtype == dtype


def test_correct_number_of_mapping_layers() -> None:
    """Tests that the layer has the correct number of mapping sublayers."""
    z_latent_size = 4
    for mapping_layers in [2, 4, 5]:
        gen_mapping = GeneratorMapping(
            z_latent_size=z_latent_size,
            mapping_layers=mapping_layers)
        # Need to build the layer to have its weights set.
        gen_mapping.build((None, z_latent_size,))
        num_dense = len([sublayer for sublayer in gen_mapping.mapping
                         if isinstance(sublayer, Dense)])
        assert num_dense == mapping_layers


def test_correct_number_of_mapping_feature_maps() -> None:
    """Tests that each sublayer has the correct number of feature maps, i.e.,
    the correct dense layer size."""
    z_latent_size = 4
    for mapping_fmaps in [20, 40, 50]:
        gen_mapping = GeneratorMapping(
            z_latent_size=z_latent_size,
            mapping_fmaps=mapping_fmaps)
        # Need to build the layer to have its weights set.
        gen_mapping.build((None, z_latent_size,))
        dense_sublayers = [sublayer for sublayer in gen_mapping.mapping
                           if isinstance(sublayer, Dense)]
        # Ignore the output dense sublayer, which has d_latent_size units.
        for sublayer in dense_sublayers[:-1]:
            if isinstance(sublayer, Dense):
                assert sublayer.units == mapping_fmaps

# TODO test trainable variables

# TODO test shapes

# TODO test conditioning labels

# TODO test normalization

# TODO test nonlinearity

# TODO test learning rate multiplier
