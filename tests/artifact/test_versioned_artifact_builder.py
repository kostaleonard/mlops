"""Tests versioned_artifact_builder.py."""

import pytest
from mlops.artifact.versioned_artifact_builder import VersionedArtifactBuilder


def test_versioned_artifact_builder_is_abstract() -> None:
    """Tests that VersionedArtifactBuilder is abstract."""
    with pytest.raises(TypeError):
        _ = VersionedArtifactBuilder()
