"""Milvus uses native schemas with ordinary Feast keys and writes."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pymilvus import DataType

from feast import Entity, FeatureView, Field
from feast.filter_models import ComparisonFilter, CompoundFilter
from feast.infra.online_stores.milvus_online_store.milvus import (
    MilvusOnlineStore,
    MilvusOnlineStoreConfig,
    _get_composite_key_name,
    _mutation_kwargs,
)
from feast.infra.online_stores.milvus_online_store.native import schema_contract
from feast.protos.feast.types.EntityKey_pb2 import EntityKey
from feast.protos.feast.types.Value_pb2 import StringList
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import RepoConfig
from feast.type_map import (
    feast_value_type_to_python_type,
    python_values_to_proto_values,
)
from feast.types import (
    Array,
    Bool,
    Bytes,
    Float32,
    Float64,
    Int32,
    Int64,
    Json,
    Map,
    ScalarMap,
    String,
    Struct,
    UnixTimestamp,
)
from feast.value_type import ValueType

NOW = datetime(2026, 1, 1, 12, 30, 0, 123456, tzinfo=timezone.utc)


def config(**options) -> RepoConfig:
    return RepoConfig(
        project="native_test",
        provider="local",
        registry="registry.db",
        entity_key_serialization_version=3,
        online_store={
            "type": "milvus",
            "partition_key": "tenant_id",
            "num_partitions": 16,
            "partition_key_isolation": False,
            "index_type": "AUTOINDEX",
            "embedding_dim": 3,
            "consistency_level": "Strong",
            "collection_consistency_level": "Strong",
            "retry_mutations": False,
            **options,
        },
    )


def view(name: str = "documents") -> FeatureView:
    analyzers = json.dumps(
        {
            "by_field": "language",
            "analyzers": {
                "default": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
                "en": {"type": "english"},
            },
        }
    )
    return FeatureView(
        name=name,
        entities=[
            Entity(name="document_id", value_type=ValueType.STRING),
            Entity(name="language", value_type=ValueType.STRING),
        ],
        schema=[
            Field(name="document_id", dtype=String),
            Field(name="language", dtype=String),
            Field(name="tenant_id", dtype=Int64, tags={"milvus.index": "INVERTED"}),
            Field(
                name="metadata",
                dtype=Json,
                tags={
                    "milvus.json_indexes": '[{"path":["arbitrary region","price"],"cast_type":"double","index_type":"AUTOINDEX"}]'
                },
            ),
            Field(name="details", dtype=Struct({"available": Bool, "count": Int32})),
            Field(name="weights", dtype=Array(Float32)),
            Field(name="flags", dtype=Array(Bool)),
            Field(
                name="labels",
                dtype=Array(String),
                tags={"max_capacity": "32", "max_length": "64"},
            ),
            Field(name="payload", dtype=Bytes),
            Field(
                name="title",
                dtype=String,
                tags={
                    "milvus.bm25": "title_sparse",
                    "milvus.multi_analyzer_params": analyzers,
                },
            ),
            Field(
                name="summary",
                dtype=String,
                tags={
                    "milvus.bm25": "summary_sparse",
                    "milvus.analyzer_params": '{"tokenizer":"standard"}',
                    "milvus.enable_match": "true",
                },
            ),
            Field(
                name="suggestion",
                dtype=String,
                tags={
                    "milvus.index": "NGRAM",
                    "milvus.index_params": '{"min_gram":2,"max_gram":3}',
                },
            ),
            Field(name="popularity", dtype=Int32, tags={"milvus.index": "STL_SORT"}),
            Field(
                name="embedding",
                dtype=Array(Float32),
                vector_index=True,
                vector_length=3,
                vector_search_metric="COSINE",
                description='{"model":"example","dimension":3}',
            ),
        ],
    )


def document() -> dict:
    return {
        "document_id": 'raw|id"1',
        "language": "en",
        "tenant_id": 9,
        "metadata": {"arbitrary region": {"price": 12.5, "tags": ["blue", "M"]}},
        "details": {"available": True, "count": 2},
        "weights": [0.0, 0.25],
        "flags": [True, False],
        "labels": ["one", "two"],
        "payload": b"\x00\xffdata",
        "title": "Blue cotton shirt",
        "summary": "Comfortable summer clothing",
        "suggestion": "blue cotton shirt",
        "popularity": 10,
        "embedding": [0.1, 0.2, 0.3],
    }


def batch(fv: FeatureView, row: dict | None = None, timestamp: datetime = NOW) -> list:
    row = document() if row is None else row
    key = EntityKey(
        join_keys=[field.name for field in fv.entity_columns],
        entity_values=[
            python_values_to_proto_values(
                [row[field.name]], field.dtype.to_value_type()
            )[0]
            for field in fv.entity_columns
        ],
    )
    values = {
        field.name: python_values_to_proto_values(
            [row[field.name]], field.dtype.to_value_type()
        )[0]
        for field in fv.features
    }
    return [(key, values, timestamp, None)]


def test_native_schema_types_dimensions_metadata_functions_indexes() -> None:
    fv, cfg, store = view(), config(), MilvusOnlineStore()
    fv.features.append(
        Field(
            name="second_vector",
            dtype=Array(Float32),
            vector_index=True,
            vector_length=5,
        )
    )
    schema, indexes = store._build_schema(cfg, fv)
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
    schema, indexes = store._build_schema(cfg, fv)
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
    next(
        field for field in fv.features if field.name == "embedding"
    ).description = "changed model"
    with pytest.raises(ValueError, match="schema drift"):
        store.ensure_schema(cfg, fv)
    store.client.create_collection.assert_not_called()
    store.client.upsert.assert_not_called()


def test_default_native_fields_and_mutation_options() -> None:
    defaults = MilvusOnlineStoreConfig()
    assert _mutation_kwargs(defaults) == {}
    with pytest.raises(ValueError, match="requires mutation_timeout=None"):
        MilvusOnlineStoreConfig(retry_mutations=False, mutation_timeout=30)
    for timeout in (0, -1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="mutation_timeout"):
            MilvusOnlineStoreConfig(mutation_timeout=timeout)
    assert _mutation_kwargs(MilvusOnlineStoreConfig(mutation_timeout=5)) == {
        "timeout": 5
    }
    assert MilvusOnlineStore().prepare_write_batch(config(), view(), batch(view()))


@pytest.mark.parametrize(
    "change",
    ["description", "analyzer", "index", "extra_index", "partitions", "isolation"],
)
def test_native_validates_actual_server_contract(change: str) -> None:
    store, cfg, fv = MilvusOnlineStore(), config(), view()
    schema, indexes = store._build_schema(cfg, fv)
    description = {
        **schema.to_dict(),
        "collection_name": store.collection_name(cfg, fv),
        "num_partitions": 16,
    }
    by_name = {index.to_dict()["index_name"]: index.to_dict() for index in indexes}
    if change == "description":
        next(field for field in description["fields"] if field["name"] == "embedding")[
            "description"
        ] = "different metadata"
    elif change == "analyzer":
        next(field for field in description["fields"] if field["name"] == "summary")[
            "params"
        ]["analyzer_params"] = {"type": "english"}
    elif change == "index":
        by_name["index_suggestion"]["min_gram"] = "4"
    elif change == "extra_index":
        by_name["unexpected_index"] = {}
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
    schema, _ = store._build_schema(cfg, fv)
    store._get_or_create_collection = MagicMock(
        return_value={"fields": schema.to_dict()["fields"]}
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
