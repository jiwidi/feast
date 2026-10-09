"""Unit test for Milvus varchar_max_length configuration."""

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pymilvus import DataType

from feast import Entity, FeatureView
from feast.field import Field
from feast.infra.online_stores.milvus_online_store.milvus import (
    MilvusOnlineStore,
    MilvusOnlineStoreConfig,
)
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


def test_varchar_max_length() -> None:
    # -- config: default and custom values ------------------------------------
    assert MilvusOnlineStoreConfig().varchar_max_length == 65535
    assert MilvusOnlineStoreConfig(varchar_max_length=1024).varchar_max_length == 1024

    # -- config: out-of-bounds values raise ValidationError -------------------
    for bad in (0, -1, 65536):
        with pytest.raises(ValidationError):
            MilvusOnlineStoreConfig(varchar_max_length=bad)

    # -- schema: configured value reaches every VARCHAR FieldSchema -----------
    entity = Entity(
        name="driver_id", join_keys=["driver_id"], value_type=ValueType.INT64
    )
    fv = FeatureView(
        name="driver_stats",
        entities=[entity],
        ttl=timedelta(days=1),
        schema=[
            Field(name="driver_id", dtype=Int64),
            Field(name="trips_today", dtype=Float32),
            Field(name="wiki_summary", dtype=String),
        ],
    )

    config = MagicMock()
    config.project = "test_project"
    config.entity_key_serialization_version = 2
    config.registry.enable_online_feature_view_versioning = False
    config.provider = "local"
    config.repo_path = None
    config.online_store = MilvusOnlineStoreConfig(varchar_max_length=4096)

    store = MilvusOnlineStore()
    schema, _ = store._build_schema(config, fv)
    for field in schema.fields:
        if field.dtype == DataType.VARCHAR:
            assert field.params["max_length"] == 4096, (
                f"field '{field.name}': got {field.params['max_length']}"
            )
