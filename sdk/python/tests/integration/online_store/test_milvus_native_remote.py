"""Native features requiring a disposable Milvus server (tested with 3.0.2).

Set FEAST_TEST_MILVUS_URI and run this file with --noconftest to avoid unrelated
integration backends. Collection names are unique and removed after each test.
"""

from __future__ import annotations

import copy
import os
import uuid

import pytest
from pymilvus import DataType, MilvusClient

from feast.filter_models import ComparisonFilter
from feast.infra.online_stores.milvus_online_store.milvus import (
    MilvusOnlineStore,
    _get_composite_key_name,
)
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.type_map import feast_value_type_to_python_type
from tests.unit.infra.online_store.test_milvus_native import (
    NOW,
    batch,
    config,
    document,
    view,
)
from tests.unit.infra.online_store.test_milvus_online_store import (
    _entity_key,
    _scalar_feature_view,
)


@pytest.fixture
def remote():
    uri = os.environ.get("FEAST_TEST_MILVUS_URI")
    if not uri:
        pytest.skip("FEAST_TEST_MILVUS_URI must point to disposable Milvus")
    cfg, fv, store = (
        config(uri=uri),
        view("native_" + uuid.uuid4().hex[:16]),
        MilvusOnlineStore(),
    )
    try:
        store.ensure_schema(cfg, fv)
        yield store, cfg, fv
    finally:
        if store.client:
            store.teardown(cfg, [fv], [])
            store.client.close()


def test_native_roundtrip_composite_keys_search_indexes_and_deletion(remote) -> None:
    store, cfg, fv = remote
    english, french = (
        document(),
        {**document(), "language": "fr", "popularity": 20, "details": None},
    )
    data = batch(fv, english) + batch(fv, french)
    store.online_write_batch(cfg, fv, data, None)
    requested = [field.name for field in fv.features]
    result = store.online_read(cfg, fv, [row[0] for row in data], requested)
    for (timestamp, values), expected in zip(result, (english, french)):
        assert timestamp == NOW
        actual = {
            key: feast_value_type_to_python_type(value) for key, value in values.items()
        }
        assert actual.pop("embedding") == pytest.approx(expected["embedding"])
        assert actual.pop("weights") == pytest.approx(expected["weights"])
        assert actual == {key: expected[key] for key in actual}
    name, client = store.collection_name(cfg, fv), store.client
    # Raw join keys, nested JSON, ARRAY and numeric scalars remain queryable.
    hits = client.query(
        name,
        filter='tenant_id == 9 and metadata["arbitrary region"]["price"] > 10 and ARRAY_CONTAINS(labels, "one") and popularity >= 10',
        output_fields=["document_id", "language"],
        consistency_level="Strong",
    )
    assert {hit["language"] for hit in hits} == {"en", "fr"}
    assert (
        len(
            client.query(
                name,
                filter='suggestion LIKE "%cotton%"',
                output_fields=["language"],
                consistency_level="Strong",
            )
        )
        == 2
    )
    for sparse, text in (("title_sparse", "cotton"), ("summary_sparse", "summer")):
        hits = client.search(
            name,
            data=[text],
            anns_field=sparse,
            search_params={"metric_type": "BM25"},
            limit=10,
            output_fields=["language"],
            consistency_level="Strong",
        )
        assert {hit["entity"]["language"] for hit in hits[0]} == {"en", "fr"}
    dense = client.search(
        name,
        data=[english["embedding"]],
        anns_field="embedding",
        search_params={"metric_type": "COSINE"},
        limit=2,
        consistency_level="Strong",
    )
    assert len(dense[0]) == 2
    # A fresh worker verifies the actual server schema and indexes.
    fresh = MilvusOnlineStore()
    try:
        fresh.ensure_schema(cfg, fv)
        store.online_delete(cfg, fv, [data[1][0]])
        assert store.online_read(cfg, fv, [data[1][0]], ["title"]) == [(None, None)]
        store.online_delete(
            cfg,
            fv,
            filters=ComparisonFilter(
                key="document_id", type="eq", value=english["document_id"]
            ),
        )
        assert store.online_read(cfg, fv, [data[0][0]], ["title"]) == [(None, None)]
        # Ordinary deletion has no retained tombstone: a caller may recreate it.
        store.online_write_batch(cfg, fv, data[:1], None)
        assert store.online_read(cfg, fv, [data[0][0]], ["title"])[0][1]
    finally:
        if fresh.client:
            fresh.client.close()


def test_native_rejects_existing_schema_drift_without_recreating_collection(
    remote,
) -> None:
    store, cfg, fv = remote
    store.online_write_batch(cfg, fv, batch(fv), None)
    changed = copy.deepcopy(fv)
    next(
        field for field in changed.features if field.name == "embedding"
    ).vector_length = 4
    fresh = MilvusOnlineStore()
    try:
        with pytest.raises(ValueError, match="schema drift"):
            fresh.ensure_schema(cfg, changed)
        assert store.online_read(cfg, fv, [batch(fv)[0][0]], ["title"])[0][1]
    finally:
        if fresh.client:
            fresh.client.close()


def test_legacy_string_schema_requires_migration_and_preserves_existing_rows() -> None:
    uri = os.environ.get("FEAST_TEST_MILVUS_URI")
    if not uri:
        pytest.skip("FEAST_TEST_MILVUS_URI must point to disposable Milvus")
    cfg = config(uri=uri, partition_key=None, num_partitions=None)
    fv = _scalar_feature_view("legacy_" + uuid.uuid4().hex[:16])
    store = MilvusOnlineStore()
    client, name = store._connect(cfg), store.collection_name(cfg, fv)
    primary = _get_composite_key_name(fv)
    # Upstream's old default stored all scalars as VARCHAR.
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(primary, DataType.VARCHAR, max_length=65535, is_primary=True)
    for field in fv.schema:
        schema.add_field(field.name, DataType.VARCHAR, max_length=65535)
    for field in ("event_ts", "created_ts"):
        schema.add_field(field, DataType.INT64)
    schema.add_field("_placeholder_vector", DataType.FLOAT_VECTOR, dim=2)
    indexes = MilvusClient.prepare_index_params()
    indexes.add_index("_placeholder_vector", index_type="FLAT", metric_type="L2")
    client.create_collection(name, schema=schema, index_params=indexes)
    data = [
        (
            _entity_key(1),
            {
                "city": ValueProto(string_val="Oslo"),
                "trips_today": ValueProto(float_val=7.0),
            },
            NOW,
            None,
        )
    ]
    original = {"city": "Paris", "trips_today": "3.0", "driver_id": "1"}
    try:
        client.upsert(
            name,
            data=[
                {
                    **original,
                    primary: store.prepare_write_batch(cfg, fv, data)[0][primary],
                    "event_ts": 0,
                    "created_ts": 0,
                    "_placeholder_vector": [0.0, 0.0],
                }
            ],
        )
        for operation in (
            lambda: store.ensure_schema(cfg, fv),
            lambda: store.online_read(cfg, fv, [_entity_key(1)], ["city"]),
            lambda: store.online_write_batch(cfg, fv, data, None),
            lambda: store.online_delete(cfg, fv, [_entity_key(1)]),
            lambda: store.online_delete(
                cfg, fv, filters=ComparisonFilter(type="eq", key="city", value="Paris")
            ),
        ):
            with pytest.raises(ValueError, match="incompatible.*rematerialize"):
                operation()
            hits = client.query(
                name,
                filter='driver_id == "1"',
                output_fields=list(original),
                consistency_level="Strong",
            )
            assert [{key: hit[key] for key in original} for hit in hits] == [original]
    finally:
        client.drop_collection(name)
        client.close()
