"""Milvus native storage format: typed schemas with ordinary Feast keys and writes."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pymilvus import DataType

from feast import Entity, FeatureStore, FeatureView, Field, FileSource, PushSource
from feast.feature_view import DUMMY_ENTITY_ID, DUMMY_ENTITY_VAL
from feast.filter_models import ComparisonFilter, CompoundFilter
from feast.infra.online_stores.milvus_online_store import native
from feast.infra.online_stores.milvus_online_store.milvus import (
    MilvusOnlineStore,
    MilvusOnlineStoreConfig,
    _get_composite_key_name,
)
from feast.infra.online_stores.milvus_online_store.native import (
    collection_schema,
    mutation_kwargs,
    schema_contract,
)
from feast.protos.feast.types.EntityKey_pb2 import EntityKey
from feast.protos.feast.types.Value_pb2 import StringList
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import RepoConfig
from feast.type_map import (
    feast_value_type_to_python_type,
)
from feast.types import (
    Array,
    Bool,
    Bytes,
    Float32,
    Float64,
    Int32,
    Json,
    Map,
    ScalarMap,
    String,
    Struct,
    UnixTimestamp,
)
from feast.value_type import ValueType
from tests.utils.milvus_native import NOW, batch, config, document, view


def test_native_schema_types_dimensions_metadata_functions_indexes() -> None:
    fv, cfg = view(), config()
    fv.features.append(
        Field(
            name="second_vector",
            dtype=Array(Float32),
            vector_index=True,
            vector_length=5,
        )
    )
    schema, indexes = collection_schema(cfg, fv)
    fields = {field.name: field for field in schema.fields}
    assert fields["metadata"].dtype == fields["details"].dtype == DataType.JSON
    assert fields["weights"].dtype == fields["flags"].dtype == DataType.ARRAY
    assert fields["flags"].element_type == DataType.BOOL
    assert fields["tenant_id"].is_partition_key and not fields["tenant_id"].nullable
    assert fields[_get_composite_key_name(fv)].is_primary
    assert fields["embedding"].params["dim"] == 3
    assert fields["second_vector"].params["dim"] == 5
    assert fields["embedding"].description == '{"model":"example","dimension":3}'
    assert len(schema.functions) == 2
    assert "_placeholder_vector" not in fields
    definitions = [index.to_dict() for index in indexes]
    assert {index["index_type"] for index in definitions} >= {
        "NGRAM",
        "STL_SORT",
        "SPARSE_INVERTED_INDEX",
    }
    assert any(
        index.get("json_path") == 'metadata["arbitrary region"]["price"]'
        for index in definitions
    )


def test_prepare_is_pure_preserves_composite_keys_and_latest_in_batch() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    data = batch(fv) + batch(fv, timestamp=NOW - timedelta(seconds=1))
    original = copy.deepcopy(data)
    rows = store.prepare_write_batch(cfg, fv, data)
    assert data == original and store.client is None
    assert len(rows) == 1
    assert rows[0]["event_ts"] == 1767270600123456
    assert rows[0]["document_id"] == document()["document_id"]
    assert rows[0]["weights"] == [0.0, 0.25]
    assert rows[0]["metadata"] == document()["metadata"]
    assert rows[0]["payload"] != document()["payload"]
    assert "title_sparse" not in rows[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding", [0.0, 0.0, 0.0]),
        ("embedding", [1.0]),
        ("embedding", [0.1, float("inf"), 0.2]),
        ("labels", ["x" * 65]),
        ("metadata", {"x": "x" * 65536}),
        ("tenant_id", None),
    ],
)
def test_invalid_native_rows_fail_before_io(field: str, value) -> None:
    store, fv, row = MilvusOnlineStore(), view(), document()
    row[field] = value
    with pytest.raises(ValueError):
        store.online_write_batch(config(), fv, batch(fv, row), None)
    assert store.client is None


def test_native_write_read_and_null_roundtrip() -> None:
    store, cfg, fv, row = MilvusOnlineStore(), config(), view(), document()
    row["details"] = None
    data = batch(fv, row)
    prepared = store.prepare_write_batch(cfg, fv, data)
    store.ensure_schema = MagicMock(return_value={"collection_name": "documents"})
    store._get_or_create_collection = store.ensure_schema
    store.client = MagicMock()
    store.client.upsert.return_value = {"upsert_count": 1}
    progress = MagicMock()
    store.online_write_batch(cfg, fv, data, progress)
    progress.assert_called_once_with(1)
    assert store.client.upsert.call_args.kwargs["retry_times"] == 0
    store.client.get.return_value = prepared
    timestamp, result = store.online_read(
        cfg, fv, [data[0][0]], ["metadata", "details", "flags", "payload"]
    )[0]
    assert timestamp == NOW
    assert {
        name: feast_value_type_to_python_type(value) for name, value in result.items()
    } == {name: row[name] for name in result}
    store.client.upsert.return_value = {"upsert_count": 0}
    progress.reset_mock()
    with pytest.raises(RuntimeError, match="acknowledge"):
        store.online_write_batch(cfg, fv, data, progress)
    progress.assert_not_called()


def test_delete_uses_same_keys_or_typed_filter_and_rejects_unbounded_delete() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    data = batch(fv)
    store.ensure_schema = MagicMock(return_value={"collection_name": "documents"})
    store.client = MagicMock()
    store.online_delete(cfg, fv, [data[0][0]])
    assert store.client.delete.call_args.kwargs["ids"] == [
        store.prepare_write_batch(cfg, fv, data)[0][_get_composite_key_name(fv)]
    ]
    store.online_delete(
        cfg, fv, filters=ComparisonFilter(key="document_id", value='x"', type="eq")
    )
    assert "filter" in store.client.delete.call_args.kwargs
    for arguments in (
        {},
        {
            "entity_keys": [],
            "filters": ComparisonFilter(key="tenant_id", value=9, type="eq"),
        },
        {"filters": CompoundFilter(type="and", filters=[])},
    ):
        with pytest.raises(ValueError):
            store.online_delete(cfg, fv, **arguments)


def test_native_schema_drift_is_rejected_before_mutating_existing_collection() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    schema, indexes = collection_schema(cfg, fv)
    description = {
        **schema.to_dict(),
        "collection_name": store.collection_name(cfg, fv),
        "num_partitions": 16,
    }
    store.client = MagicMock()
    store.client.has_collection.return_value = True
    store.client.describe_collection.return_value = description
    by_name = {index.to_dict()["index_name"]: index.to_dict() for index in indexes}
    store.client.list_indexes.return_value = list(by_name)
    store.client.describe_index.side_effect = lambda name, index: by_name[index]
    store.ensure_schema(cfg, fv)
    assert schema_contract(description) == schema_contract(schema.to_dict())
    # A changed view is validated against the server again, not against the cache.
    next(field for field in fv.features if field.name == "labels").tags[
        "max_length"
    ] = "128"
    with pytest.raises(ValueError, match="schema drift"):
        store.ensure_schema(cfg, fv)
    store.client.create_collection.assert_not_called()
    store.client.upsert.assert_not_called()


def test_default_native_fields_and_mutation_options() -> None:
    defaults = MilvusOnlineStoreConfig(storage_format="native")
    assert mutation_kwargs(defaults) == {}
    with pytest.raises(ValueError, match="requires mutation_timeout: null"):
        MilvusOnlineStoreConfig(
            storage_format="native", retry_mutations=False, mutation_timeout=30
        )
    for timeout in (0, -1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="mutation_timeout"):
            MilvusOnlineStoreConfig(storage_format="native", mutation_timeout=timeout)
    assert mutation_kwargs(
        MilvusOnlineStoreConfig(storage_format="native", mutation_timeout=5)
    ) == {"timeout": 5}
    assert MilvusOnlineStore().prepare_write_batch(config(), view(), batch(view()))


@pytest.mark.parametrize(
    "change", ["analyzer", "match", "index", "missing_index", "partitions", "isolation"]
)
def test_native_validates_actual_server_contract(change: str) -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    schema, indexes = collection_schema(cfg, fv)
    description = {
        **schema.to_dict(),
        "collection_name": store.collection_name(cfg, fv),
        "num_partitions": 16,
    }
    by_name = {index.to_dict()["index_name"]: index.to_dict() for index in indexes}
    if change == "analyzer":
        next(field for field in description["fields"] if field["name"] == "summary")[
            "params"
        ]["analyzer_params"] = {"type": "english"}
    elif change == "index":
        by_name["index_suggestion"]["min_gram"] = "4"
    elif change == "match":
        next(field for field in description["fields"] if field["name"] == "title")[
            "params"
        ]["enable_match"] = True
    elif change == "missing_index":
        del by_name["index_suggestion"]
    elif change == "partitions":
        description["num_partitions"] = 64
    else:
        description["properties"] = {"partitionkey.isolation": "true"}
    store.client = MagicMock()
    store.client.has_collection.return_value = True
    store.client.describe_collection.return_value = description
    store.client.list_indexes.return_value = list(by_name)
    store.client.describe_index.side_effect = lambda name, index: by_name[index]
    with pytest.raises(ValueError, match="drift"):
        store.ensure_schema(cfg, fv)
    store.client.create_collection.assert_not_called()
    store.client.upsert.assert_not_called()


@pytest.mark.parametrize(
    "dtype,value",
    [
        (Map, {"nested": {"flags": [True, False], "price": 12.5}}),
        (Array(Struct({"label": String})), [{"label": "one"}, {"label": "two"}]),
        (Array(Array(Int32)), [[1, 2], [], [3]]),
        (Array(Json), [{"labels": ["one"]}, {"arbitrary": None}]),
        (Array(Bytes), [b"\x00", b"\xff"]),
        (UnixTimestamp, NOW.replace(microsecond=0)),
        (Array(UnixTimestamp), [NOW.replace(microsecond=0)]),
    ],
)
def test_native_conversion_preserves_complex_protobuf_values(dtype, value) -> None:
    store, cfg, fv, row = MilvusOnlineStore(), config(), view(), document()
    fv.features.append(Field(name="extra", dtype=dtype))
    row["extra"] = value
    data = batch(fv, row)
    prepared = store.prepare_write_batch(cfg, fv, data)
    store.ensure_schema = MagicMock(return_value={"collection_name": "documents"})
    store.client = MagicMock()
    store.client.get.return_value = prepared
    result = store.online_read(cfg, fv, [data[0][0]], ["extra"])[0][1]
    # Preserve the values represented by the input protobuf, including its
    # existing null/empty semantics for nested collections.
    assert feast_value_type_to_python_type(
        result["extra"]
    ) == feast_value_type_to_python_type(data[0][1]["extra"])


def test_scalar_map_is_rejected_instead_of_changing_key_types() -> None:
    fv = view()
    fv.features.append(Field(name="numeric_keys", dtype=ScalarMap))
    with pytest.raises(ValueError, match="Unsupported native Milvus type"):
        MilvusOnlineStore().prepare_write_batch(config(), fv, [])


def test_cosine_vector_that_underflows_float32_is_rejected() -> None:
    fv, row = view(), document()
    next(field for field in fv.features if field.name == "embedding").dtype = Array(
        Float64
    )
    row["embedding"] = [1e-50, 0.0, 0.0]
    with pytest.raises(ValueError, match="Zero vector"):
        MilvusOnlineStore().prepare_write_batch(config(), fv, batch(fv, row))


@pytest.mark.parametrize(
    "value",
    ["red", "123", '{"key":"value"}', {"counter": 2**64}, {"nested": [-(2**63) - 1]}],
)
def test_unsupported_json_values_fail_before_io(value) -> None:
    store, fv = MilvusOnlineStore(), view()
    data = batch(fv)
    data[0][1]["metadata"] = ValueProto(json_val=json.dumps(value))
    with pytest.raises(ValueError, match="Top-level JSON strings|64-bit range"):
        store.online_write_batch(config(), fv, data, None)
    assert store.client is None


def test_json_array_preserves_literal_strings_through_sdk_and_feast() -> None:
    from pymilvus.client.entity_helper import convert_to_json

    store, cfg, fv = MilvusOnlineStore(), config(), view()
    data = batch(fv)
    fv.features.append(Field(name="extra", dtype=Array(Json)))
    values = ["red", "123", '{"a":1}', "", {"nested": "123"}, ["red", "123"]]
    data[0][1]["extra"] = ValueProto(
        json_list_val=StringList(val=[json.dumps(value) for value in values])
    )
    rows = store.prepare_write_batch(cfg, fv, data)
    assert json.loads(convert_to_json(rows[0]["extra"])) == values
    store.ensure_schema = MagicMock(return_value={"collection_name": "documents"})
    store.client = MagicMock()
    store.client.get.return_value = rows
    result = store.online_read(cfg, fv, [data[0][0]], ["extra"])[0][1]
    assert feast_value_type_to_python_type(result["extra"]) == values


def test_native_document_retrieval_returns_stored_values() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    data = batch(fv)
    row = store.prepare_write_batch(cfg, fv, data)[0]
    schema, _ = collection_schema(cfg, fv)
    store._get_or_create_collection = MagicMock(
        return_value={"collection_name": "documents", **schema.to_dict()}
    )
    store.client = MagicMock()
    store.client.search.return_value = [[{"entity": row, "distance": 0.9}]]
    _, key, result = store.retrieve_online_documents_v2(
        cfg, fv, ["embedding", "metadata", "flags"], [1.0, 0.0, 0.0], 1
    )[0]
    assert key == data[0][0]
    assert feast_value_type_to_python_type(result["embedding"]) == pytest.approx(
        document()["embedding"]
    )
    assert feast_value_type_to_python_type(result["metadata"]) == document()["metadata"]
    assert feast_value_type_to_python_type(result["flags"]) == document()["flags"]


def _server(store: MilvusOnlineStore, cfg: RepoConfig, fv: FeatureView) -> dict:
    """Make the mocked client describe a collection that matches the view."""
    schema, indexes = collection_schema(cfg, fv)
    description = {
        **schema.to_dict(),
        "collection_name": store.collection_name(cfg, fv),
        "num_partitions": 16,
    }
    by_name = {index.to_dict()["index_name"]: index.to_dict() for index in indexes}
    store.client = MagicMock()
    store.client.has_collection.return_value = True
    store.client.describe_collection.return_value = description
    store.client.list_indexes.return_value = list(by_name)
    store.client.describe_index.side_effect = lambda name, index: by_name[index]
    return by_name


def test_server_descriptions_extra_params_and_extra_indexes_are_compatible() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    by_name = _server(store, cfg, fv)
    description = store.client.describe_collection.return_value
    for field in description["fields"]:
        field["description"] = "server metadata"
        field.setdefault("params", {})["mmap.enabled"] = "true"
    by_name["operator_index"] = {"field_name": "popularity", "index_type": "INVERTED"}
    store.client.list_indexes.return_value = list(by_name)
    store.ensure_schema(cfg, fv)
    store.client.create_collection.assert_not_called()


def test_changed_view_revalidates_instead_of_failing_on_the_cache() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    _server(store, cfg, fv)
    store.ensure_schema(cfg, fv)
    next(field for field in fv.features if field.name == "labels").tags[
        "max_length"
    ] = "128"
    store.client.describe_collection.return_value = {
        **store.client.describe_collection.return_value,
        **collection_schema(cfg, fv)[0].to_dict(),
    }
    store.ensure_schema(cfg, fv)
    assert store.client.describe_collection.call_count == 2


def test_concurrent_create_validates_the_collection_another_writer_made() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    _server(store, cfg, fv)
    store.client.has_collection.side_effect = [False, True]
    store.client.create_collection.side_effect = RuntimeError("already exists")
    store.ensure_schema(cfg, fv)
    store.client.has_collection.side_effect = [False, False]
    store._schema_contracts.clear()
    with pytest.raises(RuntimeError, match="already exists"):
        store.ensure_schema(cfg, fv)


def test_legacy_format_rejects_native_options_and_methods() -> None:
    for option in (
        {"num_partitions": 4},
        {"partition_key_isolation": False},
        {"retry_mutations": False},
        {"mutation_timeout": 5},
    ):
        with pytest.raises(ValueError, match="require storage_format: native"):
            MilvusOnlineStoreConfig(**option)
    store, fv = MilvusOnlineStore(), view()
    cfg = RepoConfig(
        project="native_test",
        provider="local",
        registry="registry.db",
        entity_key_serialization_version=3,
        online_store={"type": "milvus"},
    )
    assert store.collection_name(cfg, fv) == "native_test_documents"
    for call in (
        lambda: store.ensure_schema(cfg, fv),
        lambda: store.prepare_write_batch(cfg, fv, batch(fv)),
        lambda: store.online_delete(cfg, fv, [batch(fv)[0][0]]),
    ):
        with pytest.raises(ValueError, match="requires storage_format: native"):
            call()
    assert store.client is None


def test_delete_by_key_validates_join_keys() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    store.ensure_schema = MagicMock(return_value={"collection_name": "documents"})
    store.client = MagicMock()
    wrong = EntityKey(
        join_keys=["document_id"], entity_values=[ValueProto(string_val="x")]
    )
    with pytest.raises(ValueError, match="join keys"):
        store.online_delete(cfg, fv, [wrong])
    store.client.delete.assert_not_called()


def test_retrieval_rejects_features_outside_the_view() -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    store.client = MagicMock()
    for feature in ("title_sparse", _get_composite_key_name(fv)):
        with pytest.raises(ValueError, match="not in documents"):
            store.retrieve_online_documents_v2(cfg, fv, [feature], [1.0, 0.0, 0.0], 1)
    store.client.search.assert_not_called()


def test_vector_index_override_does_not_inherit_store_params() -> None:
    fv = view()
    fv.features.append(
        Field(
            name="other_vector",
            dtype=Array(Float32),
            vector_index=True,
            vector_length=3,
            tags={"milvus.index": "FLAT"},
        )
    )
    cfg = config(index_type="IVF_FLAT", nlist=64)
    _, indexes = collection_schema(cfg, fv)
    definitions = {index.to_dict()["index_name"]: index.to_dict() for index in indexes}
    assert definitions["index_embedding"]["nlist"] == 64
    assert "nlist" not in definitions["index_other_vector"]


def test_array_strings_default_to_varchar_max_length() -> None:
    fv = view()
    fv.features.append(Field(name="notes", dtype=Array(String)))
    schema, _ = collection_schema(config(varchar_max_length=10000), fv)
    notes = next(field for field in schema.fields if field.name == "notes")
    assert notes.params["max_length"] == 10000


def test_equal_event_timestamps_keep_latest_created_then_last_row() -> None:
    fv, cfg = view(), config()
    first, second = document(), document()
    first["title"], second["title"] = "first", "second"
    key, first_values, _, _ = batch(fv, first)[0]
    _, second_values, _, _ = batch(fv, second)[0]
    later = NOW + timedelta(seconds=1)
    rows = native.prepare_write_batch(
        cfg, fv, [(key, first_values, NOW, later), (key, second_values, NOW, NOW)]
    )
    assert rows[0]["title"] == "first"
    rows = native.prepare_write_batch(
        cfg, fv, [(key, first_values, NOW, NOW), (key, second_values, NOW, NOW)]
    )
    assert rows[0]["title"] == "second"


def test_multi_analyzer_needs_a_supporting_sdk(monkeypatch) -> None:
    monkeypatch.setattr(native, "COMMON_TYPE_PARAMS", ("dim", "max_length"))
    with pytest.raises(ValueError, match="PyMilvus 2.5.8"):
        collection_schema(config(), view())


@pytest.mark.parametrize("entityless", [False, True])
def test_feature_store_apply_push_read_native_types(
    tmp_path: Path, entityless: bool
) -> None:
    config = RepoConfig(
        project="test_milvus",
        provider="local",
        registry=str(tmp_path / "registry.db"),
        online_store={
            "type": "milvus",
            "storage_format": "native",
            "path": str(tmp_path / "online_store.db"),
            "embedding_dim": 2,
        },
        entity_key_serialization_version=3,
        repo_path=tmp_path,
    )
    source = PushSource(
        name="updates",
        batch_source=FileSource(
            path=str(tmp_path / "unused.parquet"), timestamp_field="timestamp"
        ),
    )
    fields = [
        Field(name="amount", dtype=Float64),
        Field(name="active", dtype=Bool),
        Field(name="weights", dtype=Array(Float32)),
        Field(name="metadata", dtype=Json),
    ]
    driver = Entity(
        name="driver_id", join_keys=["driver_id"], value_type=ValueType.INT64
    )
    entities = [] if entityless else [driver]
    fv = FeatureView(
        name="native_values", entities=entities, schema=fields, source=source
    )
    store = FeatureStore(config=config)
    row = {
        "amount": 12.5,
        "active": True,
        "weights": [0.25, 0.5],
        "metadata": {"region": {"labels": ["red", "123"], "stock": 2}},
        "timestamp": datetime.now(timezone.utc),
    }
    key = {} if entityless else {"driver_id": 1}
    try:
        store.apply([*entities, source, fv])
        store.push(
            source.name,
            pd.DataFrame([{**key, **row, "metadata": json.dumps(row["metadata"])}]),
        )
        result = store.get_online_features(
            features=[f"{fv.name}:{field.name}" for field in fields],
            entity_rows=[{DUMMY_ENTITY_ID: DUMMY_ENTITY_VAL} if entityless else key],
        ).to_dict()
        for field in fields:
            assert result[field.name] == [row[field.name]]

        online = store._get_provider().online_store
        registered = store.get_feature_view(fv.name)
        schema = online.ensure_schema(config, registered)
        types = {field["name"]: field["type"] for field in schema["fields"]}
        assert types["amount"] == DataType.DOUBLE
        assert types["active"] == DataType.BOOL
        assert types["weights"] == DataType.ARRAY
        assert types["metadata"] == DataType.JSON
        if entityless:
            assert (
                online.online_read(config, registered, [EntityKey()], ["amount"])[0][1][
                    "amount"
                ].double_val
                == 12.5
            )
            online.online_delete(config, registered, [EntityKey()])
            assert online.online_read(
                config, registered, [EntityKey()], ["amount"]
            ) == [(None, None)]
            assert store.get_online_features(
                features=[f"{fv.name}:amount"],
                entity_rows=[{DUMMY_ENTITY_ID: DUMMY_ENTITY_VAL}],
            ).to_dict()["amount"] == [None]
    finally:
        online = store._get_provider().online_store
        if online.client:
            store.teardown()
            online.client.close()
