"""Tests generator_mapping.py."""

import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, ReLU, LeakyReLU
from tensorflow.keras.backend import count_params
from mlops.architecture.stylegan2.generator_mapping import GeneratorMapping
from mlops.errors import GeneratorMappingInvalidInputShapeError, \
    UnknownActivationFunctionError


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
    label_size = 2
    with pytest.raises(GeneratorMappingInvalidInputShapeError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
        # Larger input than expected.
        _ = Sequential([Input(shape=(z_latent_size + 1,)), gen_mapping])
    with pytest.raises(GeneratorMappingInvalidInputShapeError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
        # Smaller input than expected.
        _ = Sequential([Input(shape=(z_latent_size - 1,)), gen_mapping])
    with pytest.raises(GeneratorMappingInvalidInputShapeError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size,
                                       input_shape=(z_latent_size + 1,))
        # Larger input than expected, provided as layer arg.
        _ = Sequential([gen_mapping])
    with pytest.raises(GeneratorMappingInvalidInputShapeError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
        # Too few dimensions.
        _ = Sequential([Input(shape=()), gen_mapping])
    with pytest.raises(GeneratorMappingInvalidInputShapeError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size)
        # Too many dimensions.
        _ = Sequential([Input(shape=(z_latent_size, 1)), gen_mapping])
    with pytest.raises(GeneratorMappingInvalidInputShapeError):
        gen_mapping = GeneratorMapping(z_latent_size=z_latent_size,
                                       label_size=label_size)
        # Labels missing.
        _ = Sequential([Input(shape=(z_latent_size,)), gen_mapping])


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
        gen_mapping.build((None, z_latent_size))
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
        gen_mapping.build((None, z_latent_size))
        dense_sublayers = [sublayer for sublayer in gen_mapping.mapping
                           if isinstance(sublayer, Dense)]
        # Ignore the output dense sublayer, which has d_latent_size units.
        for sublayer in dense_sublayers[:-1]:
            if isinstance(sublayer, Dense):
                assert sublayer.units == mapping_fmaps


def test_correct_nonlinearity() -> None:
    """Tests that the layer uses the correct nonlinearity."""
    z_latent_size = 4
    for nonlinearity in ['relu', 'lrelu']:
        gen_mapping = GeneratorMapping(
            z_latent_size=z_latent_size,
            mapping_nonlinearity=nonlinearity)
        # Need to build the layer to have its weights set.
        gen_mapping.build((None, z_latent_size))
        if nonlinearity == 'relu':
            assert any([isinstance(sublayer, ReLU)
                        for sublayer in gen_mapping.mapping])
        else:
            assert any([isinstance(sublayer, LeakyReLU)
                        for sublayer in gen_mapping.mapping])


def test_invalid_nonlinearity_raises_error() -> None:
    """Tests that instantiating the layer with an invalid nonlinearity raises
    an error."""
    z_latent_size = 4
    gen_mapping = GeneratorMapping(z_latent_size=z_latent_size,
                                   mapping_nonlinearity='dne')
    with pytest.raises(UnknownActivationFunctionError):
        gen_mapping.build((None, z_latent_size))


def test_rms_norm_normalizes_input() -> None:
    """Tests that _rms_norm returns a normalized version of the input.
    According to the RMS Norm paper (https://arxiv.org/pdf/1910.07467.pdf),
    the output is not re-centered, but it is scaled to a sqrt(n) unit sphere,
    where n is the length of the input array."""
    batch_size = 2
    units1 = 4
    units2 = 9
    arr1 = 2 * np.ones((batch_size, units1))
    arr2 = np.array([[idx for idx in range(units2)],
                     [10 * idx for idx in range(units2)]], dtype=np.float32)
    for arr in arr1, arr2:
        normalized = GeneratorMapping._rms_norm(arr)
        normalized_vec_lengths = np.linalg.norm(normalized.numpy(), axis=1)
        assert np.isclose(
            normalized_vec_lengths,
            np.sqrt(arr.shape[1]) * np.ones_like(normalized_vec_lengths)).all()


def test_conditioning_weights_correct_shape() -> None:
    """Tests that the conditioning weights have the correct shape."""
    z_latent_size = 4
    label_size = 2
    gen_mapping = GeneratorMapping(z_latent_size=z_latent_size,
                                   label_size=label_size)
    gen_mapping.build((None, z_latent_size + label_size))
    assert gen_mapping.conditioning_weights.shape == (label_size,
                                                      z_latent_size)


def test_correct_num_trainable_variables() -> None:
    """Tests that the layer has the correct number of trainable variables
    (identical to trainable weights) when no conditioning labels are used."""
    z_latent_size = 4
    mapping_fmaps = 256
    mapping_layers = 8
    d_latent_size = 128
    model = Sequential([
        Input(shape=(z_latent_size,)),
        GeneratorMapping(z_latent_size=z_latent_size,
                         mapping_fmaps=mapping_fmaps,
                         mapping_layers=mapping_layers,
                         d_latent_size=d_latent_size)])
    first_dense_weights = (z_latent_size + 1) * mapping_fmaps
    middle_dense_weights = (mapping_fmaps + 1) * mapping_fmaps
    last_dense_weights = (mapping_fmaps + 1) * d_latent_size
    expected_weights = first_dense_weights + \
        (mapping_layers - 2) * middle_dense_weights + \
        last_dense_weights
    trainable_variable_sum = sum(
        count_params(variables) for variables in model.trainable_variables)
    assert expected_weights == trainable_variable_sum


def test_correct_num_trainable_variables_with_conditioning() -> None:
    """Tests that the layer has the correct number of trainable variables
    (identical to trainable weights) when no conditioning labels are used."""
    z_latent_size = 4
    label_size = 8
    mapping_fmaps = 256
    mapping_layers = 8
    d_latent_size = 128
    model = Sequential([
        Input(shape=(z_latent_size + label_size,)),
        GeneratorMapping(z_latent_size=z_latent_size,
                         label_size=label_size,
                         mapping_fmaps=mapping_fmaps,
                         mapping_layers=mapping_layers,
                         d_latent_size=d_latent_size)])
    conditioning_weights = z_latent_size * label_size
    first_dense_weights = (2 * z_latent_size + 1) * mapping_fmaps
    middle_dense_weights = (mapping_fmaps + 1) * mapping_fmaps
    last_dense_weights = (mapping_fmaps + 1) * d_latent_size
    expected_weights = conditioning_weights + \
        first_dense_weights + \
        (mapping_layers - 2) * middle_dense_weights + \
        last_dense_weights
    trainable_variable_sum = sum(
        count_params(variables) for variables in model.trainable_variables)
    assert expected_weights == trainable_variable_sum
