"""Shared feature view and rows for Milvus native storage tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from feast import Entity, FeatureView, Field
from feast.protos.feast.types.EntityKey_pb2 import EntityKey
from feast.repo_config import RepoConfig
from feast.type_map import python_values_to_proto_values
from feast.types import (
    Array,
    Bool,
    Bytes,
    Float32,
    Int32,
    Int64,
    Json,
    String,
    Struct,
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
            "storage_format": "native",
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
                tags={"milvus.max_capacity": "32", "max_length": "64"},
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
