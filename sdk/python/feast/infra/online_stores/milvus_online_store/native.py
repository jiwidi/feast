"""Native Milvus storage: typed fields, declarative search indexes and deletion.

Used when the online store config sets ``storage_format: native``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
import struct
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from pymilvus import CollectionSchema, DataType, Function, FunctionType, MilvusClient
from pymilvus.orm.constants import COMMON_TYPE_PARAMS

from feast import FeatureView, Field
from feast.feature_view import (
    DUMMY_ENTITY_FIELD,
    DUMMY_ENTITY_ID,
    DUMMY_ENTITY_NAME,
    DUMMY_ENTITY_VAL,
)
from feast.filter_models import ComparisonFilter, CompoundFilter
from feast.infra.key_encoding_utils import deserialize_entity_key, serialize_entity_key
from feast.infra.online_stores.milvus_online_store import milvus as base
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import RepoConfig
from feast.type_map import (
    feast_value_type_to_python_type,
    python_values_to_proto_values,
)
from feast.types import Array, Float32, Float64, PrimitiveFeastType
from feast.value_type import ValueType

if TYPE_CHECKING:
    from feast.infra.online_stores.milvus_online_store.milvus import (
        MilvusOnlineStore,
        MilvusOnlineStoreConfig,
    )


SCALARS = {
    ValueType.STRING: DataType.VARCHAR,
    ValueType.BYTES: DataType.VARCHAR,
    ValueType.IMAGE_BYTES: DataType.VARCHAR,
    ValueType.BOOL: DataType.BOOL,
    ValueType.INT32: DataType.INT32,
    ValueType.INT64: DataType.INT64,
    ValueType.UNIX_TIMESTAMP: DataType.INT64,
    ValueType.FLOAT: DataType.FLOAT,
    ValueType.DOUBLE: DataType.DOUBLE,
}
JSON_TYPES = {ValueType.JSON, ValueType.STRUCT, ValueType.MAP}
INTERNAL_FIELDS = {"event_ts", "created_ts"}
PLACEHOLDER = "_placeholder_vector"


def boolean_tag(tags: dict[str, str], name: str, default: bool) -> bool:
    value = tags.get(name, str(default)).lower()
    if value not in ("true", "false"):
        raise ValueError(f"{name} must be 'true' or 'false'")
    return value == "true"


def _json_tag(field: Field, tag: str, expected: type, default: str) -> Any:
    try:
        value = json.loads(field.tags.get(tag, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field.name}.{tag} must contain valid JSON") from error
    if not isinstance(value, expected):
        raise ValueError(f"{field.name}.{tag} must contain a JSON {expected.__name__}")
    return value


def _limit(field: Field, tag: str, default: int, maximum: int) -> int:
    value = int(field.tags.get(tag, default))
    if not 1 <= value <= maximum:
        raise ValueError(f"{field.name}.{tag} must be between 1 and {maximum}")
    return value


def field_spec(field: Field, config: MilvusOnlineStoreConfig) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "field_name": field.name,
        "description": field.description or "",
        "nullable": boolean_tag(field.tags, "milvus.nullable", not field.vector_index),
    }
    if field.vector_index:
        if not isinstance(field.dtype, Array) or field.dtype.base_type not in (
            Float32,
            Float64,
        ):
            raise ValueError(f"{field.name}: indexed vectors must be float arrays")
        dimension = field.vector_length or config.embedding_dim
        if dimension is None or not 2 <= dimension <= 32768:
            raise ValueError(
                f"{field.name}: vector dimension must be between 2 and 32768"
            )
        if spec["nullable"]:
            raise ValueError(f"{field.name}: indexed vectors cannot be nullable")
        spec.update(datatype=DataType.FLOAT_VECTOR, dim=dimension)
    elif isinstance(field.dtype, Array):
        base = field.dtype.base_type
        if isinstance(base, Array) or base.to_value_type() in JSON_TYPES:
            spec["datatype"] = DataType.JSON
        elif base.to_value_type() in SCALARS:
            spec.update(
                datatype=DataType.ARRAY,
                element_type=SCALARS[base.to_value_type()],
                max_capacity=_limit(field, "milvus.max_capacity", 4096, 4096),
            )
            if spec["element_type"] == DataType.VARCHAR:
                spec["max_length"] = _limit(
                    field, "max_length", config.varchar_max_length or 65535, 65535
                )
        else:
            raise ValueError(f"Unsupported native Milvus array type for {field.name}")
    elif field.dtype.to_value_type() in JSON_TYPES:
        spec["datatype"] = DataType.JSON
    elif field.dtype.to_value_type() in SCALARS:
        spec["datatype"] = SCALARS[field.dtype.to_value_type()]
        if spec["datatype"] == DataType.VARCHAR:
            spec["max_length"] = _limit(
                field, "max_length", config.varchar_max_length or 65535, 65535
            )
    else:
        raise ValueError(f"Unsupported native Milvus type for {field.name}")
    analyzer_tags = ("milvus.analyzer_params", "milvus.multi_analyzer_params")
    if all(tag in field.tags for tag in analyzer_tags):
        raise ValueError(f"{field.name}: choose one analyzer configuration")
    if (
        analyzer_tags[1] in field.tags
        and "multi_analyzer_params" not in COMMON_TYPE_PARAMS
    ):
        raise ValueError(
            f"{field.name}: milvus.multi_analyzer_params requires PyMilvus 2.5.8 or newer"
        )
    if (
        field.tags.get("milvus.bm25")
        or any(tag in field.tags for tag in analyzer_tags)
        or boolean_tag(field.tags, "milvus.enable_match", False)
    ):
        if (
            spec["datatype"] != DataType.VARCHAR
            or field.dtype.to_value_type() != ValueType.STRING
        ):
            raise ValueError(f"{field.name}: text analysis requires a String field")
        spec["enable_analyzer"] = True
        for tag in analyzer_tags:
            if tag in field.tags:
                spec[tag.removeprefix("milvus.")] = _json_tag(field, tag, dict, "{}")
        if boolean_tag(field.tags, "milvus.enable_match", False):
            if analyzer_tags[1] in field.tags:
                raise ValueError(
                    "Milvus does not support enable_match with multiple analyzers"
                )
            spec["enable_match"] = True
    return spec


def build_schema(
    table: FeatureView,
    config: MilvusOnlineStoreConfig,
    primary_key: str,
    partition_key: str | None,
    vector_index_params: dict[str, Any],
) -> tuple[CollectionSchema, Any]:
    """Build one declarative schema without connecting to Milvus."""
    if not table.entity_columns:
        raise ValueError(
            "Milvus requires a registered/resolved FeatureView with entity_columns"
        )
    fields = {field.name: field for field in table.schema}
    reserved = INTERNAL_FIELDS | {primary_key, PLACEHOLDER}
    if reserved.intersection(fields):
        raise ValueError(
            f"Reserved Milvus fields: {sorted(reserved.intersection(fields))}"
        )
    if len(fields) != len(table.schema):
        raise ValueError("Duplicate field names in Milvus schema")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    indexes = MilvusClient.prepare_index_params()
    schema.add_field(
        primary_key,
        DataType.VARCHAR,
        max_length=config.varchar_max_length or 65535,
        is_primary=True,
    )
    for name in sorted(INTERNAL_FIELDS):
        schema.add_field(name, DataType.INT64)
    entity_names = {field.name for field in table.entity_columns}
    outputs = [
        field.tags["milvus.bm25"]
        for field in fields.values()
        if field.tags.get("milvus.bm25")
    ]
    if len(outputs) != len(set(outputs)) or set(outputs).intersection(
        set(fields) | reserved
    ):
        raise ValueError("BM25 output names must be unique and cannot shadow fields")
    for field in fields.values():
        spec = field_spec(field, config)
        if field.name in entity_names or field.name == partition_key:
            if field.tags.get("milvus.nullable", "false").lower() != "false":
                raise ValueError(
                    f"{field.name}: entity and partition fields cannot be nullable"
                )
            spec["nullable"] = False
        if field.name == partition_key:
            if spec["datatype"] not in (DataType.VARCHAR, DataType.INT64):
                raise ValueError("Milvus partition keys must be String or Int64")
            spec["is_partition_key"] = True
        multi = spec.get("multi_analyzer_params")
        if multi and (
            multi.get("by_field") not in fields
            or "default" not in multi.get("analyzers", {})
        ):
            raise ValueError(
                f"{field.name}: multi-analyzer needs a schema by_field and default analyzer"
            )
        schema.add_field(**spec)
        index_type = (
            field.tags.get(
                "milvus.index", config.index_type if field.vector_index else "false"
            )
            or "FLAT"
        )
        if index_type.lower() == "true":
            index_type = "INVERTED"
        if field.vector_index and index_type.lower() == "false":
            raise ValueError(f"{field.name}: Milvus requires an index for every vector")
        if index_type.lower() != "false":
            options: dict[str, Any] = {"index_type": index_type.upper()}
            params = _json_tag(field, "milvus.index_params", dict, "{}")
            if field.vector_index:
                options["metric_type"] = (
                    field.vector_search_metric or config.metric_type or "COSINE"
                ).upper()
                if not {"milvus.index", "milvus.index_params"} & set(field.tags):
                    params = vector_index_params
            options["params"] = params
            indexes.add_index(field.name, index_name=f"index_{field.name}", **options)
        elif "milvus.index_params" in field.tags:
            raise ValueError(f"{field.name}: index_params requires an index")
        for position, path in enumerate(
            _json_tag(field, "milvus.json_indexes", list, "[]")
        ):
            if spec["datatype"] != DataType.JSON or not isinstance(path, dict):
                raise ValueError(
                    f"{field.name}: json_indexes requires JSON path objects"
                )
            keys = path.get("path")
            if (
                not isinstance(keys, list)
                or not keys
                or not all(isinstance(key, str) for key in keys)
                or not isinstance(path.get("cast_type"), str)
            ):
                raise ValueError(f"{field.name}: invalid JSON index path or cast_type")
            indexes.add_index(
                field.name,
                index_name=f"{field.name}_path_{position}",
                index_type=path.get("index_type", "INVERTED"),
                params={
                    "json_path": field.name
                    + "".join(
                        f"[{json.dumps(key, ensure_ascii=False)}]" for key in keys
                    ),
                    "json_cast_type": path["cast_type"],
                },
            )
        output = field.tags.get("milvus.bm25")
        if output:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", output):
                raise ValueError(f"{field.name}: invalid BM25 output identifier")
            schema.add_field(output, DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name=f"{output}_fn",
                    input_field_names=[field.name],
                    output_field_names=[output],
                    function_type=FunctionType.BM25,
                )
            )
            indexes.add_index(
                output,
                index_name=f"index_{output}",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )
    if partition_key and partition_key not in fields:
        raise ValueError(f"Partition key {partition_key!r} is not in the feature view")
    if not outputs and not any(field.vector_index for field in fields.values()):
        schema.add_field(PLACEHOLDER, DataType.FLOAT_VECTOR, dim=2)
        indexes.add_index(
            PLACEHOLDER,
            index_name=f"index_{PLACEHOLDER}",
            index_type="FLAT",
            metric_type="L2",
        )
    if len(schema.fields) > 64:
        raise ValueError(
            "Milvus schema exceeds 64 fields including keys, timestamps and BM25 outputs"
        )
    schema.verify()
    return schema, indexes


# Params that must match in both directions. Other server params, such as mmap
# settings, only need to match when the feature view declares them.
STRICT_PARAMS = (
    "dim",
    "max_length",
    "max_capacity",
    "enable_analyzer",
    "analyzer_params",
    "multi_analyzer_params",
    "enable_match",
)
BOOLEAN_PARAMS = ("enable_analyzer", "enable_match")


def schema_contract(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize SDK and server schemas. Field descriptions are not compared."""
    fields = {}
    for field in schema["fields"]:
        params = dict(field.get("params", {}))
        for name in ("analyzer_params", "multi_analyzer_params"):
            if isinstance(params.get(name), str):
                params[name] = json.loads(params[name])
        for name in ("dim", "max_length", "max_capacity"):
            if name in params:
                params[name] = int(params[name])
        for name in BOOLEAN_PARAMS:
            params[name] = str(params.get(name, False)).lower() == "true"
        fields[field["name"]] = {
            "type": int(field["type"]),
            "element_type": int(field.get("element_type", 0)),
            "params": params,
            **{
                name: field.get(name, False)
                for name in (
                    "nullable",
                    "is_primary",
                    "is_partition_key",
                    "is_function_output",
                )
            },
        }
    functions = [
        {
            name: function.get(name, {} if name == "params" else "")
            for name in (
                "name",
                "type",
                "input_field_names",
                "output_field_names",
                "params",
            )
        }
        for function in schema.get("functions", [])
    ]
    return {
        "fields": fields,
        "functions": sorted(functions, key=lambda function: function["name"]),
        "auto_id": schema.get("auto_id", False),
        "enable_dynamic_field": schema.get("enable_dynamic_field", False),
    }


def schema_drift(expected: dict[str, Any], actual: dict[str, Any]) -> Optional[str]:
    """Return the first incompatible part of a server schema, or None."""
    want, have = schema_contract(expected), schema_contract(actual)
    if want["fields"].keys() != have["fields"].keys():
        return "field set"
    for name, field in want["fields"].items():
        server = have["fields"][name]
        wanted_params, server_params = field["params"], server["params"]
        if (
            {key: value for key, value in field.items() if key != "params"}
            != {key: value for key, value in server.items() if key != "params"}
            or any(
                wanted_params.get(key) != server_params.get(key)
                for key in STRICT_PARAMS
            )
            or any(
                server_params.get(key) != value for key, value in wanted_params.items()
            )
        ):
            return f"field {name}"
    for part in ("functions", "auto_id", "enable_dynamic_field"):
        if want[part] != have[part]:
            return part
    return None


def _scalar(name: str, value: Any, dtype: DataType, max_length: int = 65535) -> None:
    valid = True
    if dtype == DataType.VARCHAR:
        valid = isinstance(value, str) and len(value.encode()) <= max_length
    elif dtype == DataType.BOOL:
        valid = isinstance(value, bool)
    elif dtype in (DataType.INT32, DataType.INT64):
        bits = 32 if dtype == DataType.INT32 else 64
        valid = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and -(2 ** (bits - 1)) <= value < 2 ** (bits - 1)
        )
    elif dtype in (DataType.FLOAT, DataType.DOUBLE):
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        if valid and dtype == DataType.FLOAT:
            valid = abs(value) <= 3.4028234663852886e38
    if not valid:
        raise ValueError(f"Invalid or out-of-range {dtype.name} value for {name}")


def convert_value(field: Field, value: Any, *, to_storage: bool) -> Any:
    """Only bytes and timestamp scalars need a reversible storage conversion."""
    if value is None:
        return None
    if isinstance(field.dtype, Array):
        if not isinstance(value, list):
            raise ValueError(f"{field.name}: expected an array")
        if not isinstance(
            field.dtype.base_type, Array
        ) and field.dtype.base_type.to_value_type() not in (
            ValueType.BYTES,
            ValueType.IMAGE_BYTES,
            ValueType.UNIX_TIMESTAMP,
        ):
            return value
        element = Field(name=field.name, dtype=field.dtype.base_type)
        return [convert_value(element, item, to_storage=to_storage) for item in value]
    dtype = field.dtype.to_value_type()
    if dtype in (ValueType.BYTES, ValueType.IMAGE_BYTES):
        if to_storage:
            if not isinstance(value, bytes):
                raise ValueError(f"{field.name}: expected bytes")
            return base64.b64encode(value).decode("ascii")
        return base64.b64decode(value)
    if dtype == ValueType.UNIX_TIMESTAMP:
        if to_storage:
            if not isinstance(value, datetime):
                raise ValueError(f"{field.name}: expected a timestamp")
            return int(value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp())
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return value


def validate_value(field: Field, spec: dict[str, Any], value: Any, metric: str) -> None:
    if value is None:
        if not spec.get("nullable", False):
            raise ValueError(f"Required Milvus field {field.name} is null or missing")
        return
    dtype = spec["datatype"]
    if dtype == DataType.FLOAT_VECTOR:
        if not isinstance(value, list) or len(value) != spec["dim"]:
            raise ValueError(f"Invalid vector dimension for {field.name}")
        for element in value:
            _scalar(field.name, element, DataType.FLOAT)
        # FLOAT_VECTOR stores float32 even when the Feast input is Float64.
        if metric.upper() == "COSINE" and not any(
            struct.unpack("!f", struct.pack("!f", float(element)))[0]
            for element in value
        ):
            raise ValueError(f"Zero vector is invalid for COSINE field {field.name}")
    elif dtype == DataType.JSON:
        if isinstance(value, str):
            raise ValueError(
                f"Top-level JSON strings are not supported for {field.name}"
            )
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise ValueError(f"{field.name}: JSON keys must be strings")
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, int) and not -(2**63) <= item < 2**64:
                raise ValueError(
                    f"{field.name}: JSON integer exceeds the SDK's 64-bit range"
                )
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError(f"{field.name} is not finite JSON data") from error
        if len(encoded) > 65536:
            raise ValueError(f"{field.name} exceeds Milvus's 65536-byte JSON limit")
    elif dtype == DataType.ARRAY:
        if not isinstance(value, list) or len(value) > spec["max_capacity"]:
            raise ValueError(f"Invalid or oversized array {field.name}")
        for element in value:
            _scalar(
                field.name, element, spec["element_type"], spec.get("max_length", 65535)
            )
    else:
        _scalar(field.name, value, dtype, spec.get("max_length", 65535))


def to_proto(field: Field, value: Any) -> ValueProto:
    """Keep literal JSON strings distinct from already serialized JSON input."""
    value = convert_value(field, value, to_storage=False)
    dtype = field.dtype.to_value_type()
    if value is not None:
        if dtype == ValueType.JSON:
            value = json.dumps(value, ensure_ascii=False, allow_nan=False)
        elif dtype == ValueType.JSON_LIST:
            value = [
                json.dumps(item, ensure_ascii=False, allow_nan=False) for item in value
            ]
    return python_values_to_proto_values([value], dtype)[0]


WriteBatch = list[
    tuple[EntityKeyProto, dict[str, ValueProto], datetime, Optional[datetime]]
]
Document = tuple[
    Optional[datetime], Optional[EntityKeyProto], Optional[dict[str, ValueProto]]
]


def resolve_view(table: FeatureView) -> FeatureView:
    """Feast hides its dummy entity on views passed to push/write ingestion."""
    if not table.entity_columns and (
        not table.entities or table.entities == [DUMMY_ENTITY_NAME]
    ):
        table = copy.copy(table)
        table.entities = [DUMMY_ENTITY_NAME]
        table.entity_columns = [DUMMY_ENTITY_FIELD]
    return table


def _entity_key(table: FeatureView, key: EntityKeyProto) -> EntityKeyProto:
    if (
        table.join_keys == [DUMMY_ENTITY_ID]
        and not key.join_keys
        and not key.entity_values
    ):
        return EntityKeyProto(
            join_keys=[DUMMY_ENTITY_ID],
            entity_values=[ValueProto(string_val=DUMMY_ENTITY_VAL)],
        )
    return key


def _checked_entity_key(table: FeatureView, key: EntityKeyProto) -> EntityKeyProto:
    key = _entity_key(table, key)
    entities = set(table.join_keys)
    if (
        len(key.join_keys) != len(key.entity_values)
        or set(key.join_keys) != entities
        or len(key.join_keys) != len(entities)
    ):
        raise ValueError(
            "Entity key does not match the registered feature view join keys"
        )
    return key


def _key_id(config: RepoConfig, key: EntityKeyProto) -> str:
    return serialize_entity_key(
        key, entity_key_serialization_version=config.entity_key_serialization_version
    ).hex()


def _vector_index_params(online: MilvusOnlineStoreConfig) -> dict[str, Any]:
    """FLAT and AUTOINDEX need no build parameters by default."""
    if online.index_params is not None:
        return dict(online.index_params)
    if (online.index_type or "FLAT").upper() in ("FLAT", "AUTOINDEX"):
        return {}
    return {"nlist": online.nlist}


def mutation_kwargs(online: MilvusOnlineStoreConfig) -> dict[str, Any]:
    if not online.retry_mutations:
        return {"timeout": None, "retry_times": 0, "retry_on_rate_limit": False}
    if online.mutation_timeout is not None:
        return {"timeout": online.mutation_timeout}
    return {}


def collection_schema(
    config: RepoConfig, table: FeatureView
) -> tuple[CollectionSchema, Any]:
    table = resolve_view(table)
    return build_schema(
        table,
        config.online_store,
        base._get_composite_key_name(table),
        base._partition_key_name(config.online_store, table),
        _vector_index_params(config.online_store),
    )


def _schema_mismatch(collection: str, detail: str) -> ValueError:
    return ValueError(
        f"Milvus schema drift for collection {collection!r} ({detail}). "
        "The existing collection is incompatible with this feature view. "
        "Create a new feature view or feature-view version and rematerialize "
        "before switching readers. Feast will not alter or delete the existing collection."
    )


def ensure_collection(
    store: MilvusOnlineStore, config: RepoConfig, table: FeatureView
) -> dict[str, Any]:
    """Create a missing collection, validate an existing one and load it."""
    online = config.online_store
    schema, indexes = collection_schema(config, table)
    partitioned = base._partition_key_name(online, resolve_view(table)) is not None
    name = store.collection_name(config, table)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema": schema_contract(schema.to_dict()),
                "indexes": [index.to_dict() for index in indexes],
                "num_partitions": online.num_partitions,
                "partition_key_isolation": online.partition_key_isolation,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if store._schema_contracts.get(name) == fingerprint:
        return store._collections[name]
    client = store._connect(config)
    created = not client.has_collection(name)
    if created:
        options = base._collection_consistency_kwargs(online)
        if partitioned:
            if online.num_partitions is not None:
                options["num_partitions"] = online.num_partitions
            if online.partition_key_isolation is not None:
                options["properties"] = {
                    "partitionkey.isolation": str(
                        online.partition_key_isolation
                    ).lower()
                }
        try:
            client.create_collection(
                collection_name=name, schema=schema, index_params=indexes, **options
            )
        except Exception:
            # Another writer can create the collection first. Validate it below.
            if not client.has_collection(name):
                raise
            created = False
    description = client.describe_collection(name)
    drift = schema_drift(schema.to_dict(), description)
    if drift:
        raise _schema_mismatch(name, drift)
    if partitioned:
        if (
            online.num_partitions is not None
            and description.get("num_partitions") != online.num_partitions
        ):
            raise _schema_mismatch(name, "partition count")
        properties = description.get("properties", {})
        if isinstance(properties, list):
            properties = {entry["key"]: entry["value"] for entry in properties}
        isolated = (
            str(properties.get("partitionkey.isolation", "false")).lower() == "true"
        )
        if (
            online.partition_key_isolation is not None
            and isolated != online.partition_key_isolation
        ):
            raise _schema_mismatch(name, "partition isolation")
    actual_indexes = [
        client.describe_index(name, index_name)
        for index_name in client.list_indexes(name)
    ]
    for index in indexes:
        expected = index.to_dict()
        index_name = expected.pop("index_name") or expected["field_name"]
        # Milvus Lite names an index after its field and ignores the given name,
        # so match on the declared field, type, metric and parameters. Extra
        # indexes that an operator added are allowed.
        if not any(
            all(str(actual.get(key)) == str(value) for key, value in expected.items())
            for actual in actual_indexes
        ):
            raise _schema_mismatch(name, f"index {index_name}")
    if not created:
        store._ensure_loaded(name)
    store._collections[name] = description
    store._schema_contracts[name] = fingerprint
    return description


def prepare_write_batch(
    config: RepoConfig, table: FeatureView, data: WriteBatch
) -> list[dict[str, Any]]:
    """Validate and convert a batch without I/O or changing inputs.

    This does not compare against stored rows. Within the batch, the row with the
    latest event timestamp wins, then the latest created timestamp, then the last row.
    """
    table = resolve_view(table)
    schema, _ = collection_schema(config, table)
    specs = {field.name: field.to_dict() for field in schema.fields}
    primary = base._get_composite_key_name(table)
    fields = {field.name: field for field in table.schema}
    metric = config.online_store.metric_type or "COSINE"
    rows: dict[str, dict[str, Any]] = {}
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def micros(value: datetime) -> int:
        delta = value.replace(tzinfo=value.tzinfo or timezone.utc) - epoch
        return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds

    for entity_key, values, timestamp, created in data:
        entity_key = _checked_entity_key(table, entity_key)
        unexpected = set(values) - set(fields)
        if unexpected:
            raise ValueError(f"Unknown Milvus feature fields: {sorted(unexpected)}")
        row = {
            name: feast_value_type_to_python_type(value)
            for name, value in values.items()
        }
        for name, value in zip(entity_key.join_keys, entity_key.entity_values):
            decoded = feast_value_type_to_python_type(value)
            if name in row and row[name] != decoded:
                raise ValueError(f"Feature value disagrees with entity join key {name}")
            row[name] = decoded
        for name, field in fields.items():
            spec = specs[name]
            spec = {**spec.get("params", {}), **spec, "datatype": spec["type"]}
            row[name] = convert_value(field, row.get(name), to_storage=True)
            validate_value(field, spec, row[name], field.vector_search_metric or metric)
        key = _key_id(config, entity_key)
        if len(key) > (config.online_store.varchar_max_length or 65535):
            raise ValueError("Serialized entity key exceeds varchar_max_length")
        row.update(
            {
                primary: key,
                "event_ts": micros(timestamp),
                "created_ts": micros(created) if created else 0,
            }
        )
        if PLACEHOLDER in specs:
            row[PLACEHOLDER] = [0.0, 0.0]
        previous = rows.get(key)
        if previous is None or (previous["event_ts"], previous["created_ts"]) <= (
            row["event_ts"],
            row["created_ts"],
        ):
            rows[key] = row
    return list(rows.values())


def write_batch(
    store: MilvusOnlineStore,
    config: RepoConfig,
    table: FeatureView,
    data: WriteBatch,
    progress: Optional[Callable[[int], Any]],
) -> None:
    rows = prepare_write_batch(config, table, data)
    if not rows:
        return
    collection = store.ensure_schema(config, table)
    result = store._connect(config).upsert(
        collection_name=collection["collection_name"],
        data=rows,
        **mutation_kwargs(config.online_store),
    )
    if result.get("upsert_count") != len(rows):
        raise RuntimeError("Milvus did not acknowledge every row in the batch")
    if progress:
        progress(len(data))


def read(
    store: MilvusOnlineStore,
    config: RepoConfig,
    table: FeatureView,
    entity_keys: list[EntityKeyProto],
    requested_features: Optional[list[str]],
) -> list[tuple[Optional[datetime], Optional[dict[str, ValueProto]]]]:
    if not entity_keys:
        return []
    table = resolve_view(table)
    fields = {field.name: field for field in table.schema}
    requested = (
        requested_features
        if requested_features is not None
        else [field.name for field in table.features]
    )
    unknown = set(requested) - set(fields)
    if unknown:
        raise ValueError(
            f"Requested features are not in {table.name}: {sorted(unknown)}"
        )
    collection = store.ensure_schema(config, table)
    primary = base._get_composite_key_name(table)
    keys = [_key_id(config, _entity_key(table, key)) for key in entity_keys]
    hits = store._connect(config).get(
        collection_name=collection["collection_name"],
        ids=keys,
        output_fields=[primary, "event_ts", *requested],
        **base._consistency_kwargs(config.online_store),
    )
    by_key = {hit[primary]: hit for hit in hits}
    result: list[tuple[Optional[datetime], Optional[dict[str, ValueProto]]]] = []
    for key in keys:
        hit = by_key.get(key)
        if hit is None:
            result.append((None, None))
            continue
        values = {name: to_proto(fields[name], hit.get(name)) for name in requested}
        result.append(
            (datetime.fromtimestamp(hit["event_ts"] / 1e6, timezone.utc), values)
        )
    return result


def delete(
    store: MilvusOnlineStore,
    config: RepoConfig,
    table: FeatureView,
    entity_keys: Optional[list[EntityKeyProto]],
    filters: Optional[Union[ComparisonFilter, CompoundFilter]],
) -> None:
    if (entity_keys is None) == (filters is None):
        raise ValueError("Provide exactly one of entity_keys or filters")
    if entity_keys == []:
        return
    table = resolve_view(table)
    selector: dict[str, Any]
    if entity_keys is not None:
        selector = {
            "ids": [
                _key_id(config, _checked_entity_key(table, key)) for key in entity_keys
            ]
        }
    else:
        expression = base.MilvusFilterTranslator().translate(filters)
        if not expression:
            raise ValueError("Deletion filter must not be empty")
        selector = {"filter": expression}
    collection = store.ensure_schema(config, table)
    store._connect(config).delete(
        collection_name=collection["collection_name"],
        **selector,
        **mutation_kwargs(config.online_store),
    )


def retrieve_documents(
    store: MilvusOnlineStore,
    config: RepoConfig,
    table: FeatureView,
    requested_features: list[str],
    embedding: Optional[list[float]],
    top_k: int,
    distance_metric: Optional[str],
    query_string: Optional[str],
    filters: Optional[Union[ComparisonFilter, CompoundFilter]],
) -> list[Document]:
    """Vector search, LIKE text search, or both, with the same semantics as legacy."""
    table = resolve_view(table)
    online = config.online_store
    fields = {field.name: field for field in table.schema}
    requested = requested_features or []
    unknown = set(requested) - set(fields)
    if unknown:
        raise ValueError(
            f"Requested features are not in {table.name}: {sorted(unknown)}"
        )
    collection = store.ensure_schema(config, table)
    if not online.vector_enabled:
        raise ValueError("Vector search is not enabled in the online store config")
    if embedding is None and query_string is None:
        raise ValueError("Either embedding or query_string must be provided")
    primary = base._get_composite_key_name(table)
    output_fields = [primary, *requested, "created_ts", "event_ts"]
    metadata_filter = base.MilvusFilterTranslator().translate(filters)
    text_filter = None
    if query_string is not None:
        strings = [
            field.name
            for field in table.features
            if isinstance(field.dtype, PrimitiveFeastType)
            and field.dtype.to_value_type() == ValueType.STRING
        ]
        if not strings:
            raise ValueError(
                "No string fields found in the feature view for text search"
            )
        escaped = base._milvus_escape_string(query_string)
        text_filter = " OR ".join(
            f"{name} LIKE '%{escaped}%'" for name in strings if name in requested
        )
        if not text_filter and embedding is None:
            raise ValueError("No text fields found in requested features for search")
    parts = [part for part in (text_filter, metadata_filter) if part]
    expression = (
        parts[0] if len(parts) == 1 else " and ".join(f"({part})" for part in parts)
    )
    client = store._connect(config)
    if embedding is not None:
        results = client.search(
            collection_name=collection["collection_name"],
            data=[embedding],
            anns_field=next(
                (
                    field["name"]
                    for field in collection["fields"]
                    if field["type"] == DataType.FLOAT_VECTOR
                    and field["name"] in output_fields
                ),
                None,
            ),
            search_params={
                "metric_type": distance_metric or online.metric_type,
                "params": base._search_params(online),
            },
            limit=top_k,
            output_fields=output_fields,
            filter=expression or None,
            **base._consistency_kwargs(online),
        )
    else:
        results = [
            [{"entity": entity, "distance": -1.0}]
            for entity in client.query(
                collection_name=collection["collection_name"],
                filter=expression,
                output_fields=output_fields,
                limit=top_k,
                **base._consistency_kwargs(online),
            )
        ]
    documents: list[Document] = []
    for hits in results:
        for hit in hits:
            entity = hit.get("entity", {})
            raw_key = entity.get(primary)
            event_ts = entity.get("event_ts")
            values = {
                name: to_proto(fields[name], entity.get(name)) for name in requested
            }
            distance = hit.get("distance")
            values["distance"] = (
                ValueProto(float_val=distance) if distance is not None else ValueProto()
            )
            documents.append(
                (
                    datetime.fromtimestamp(event_ts / 1e6, timezone.utc)
                    if event_ts is not None
                    else None,
                    deserialize_entity_key(
                        bytes.fromhex(raw_key),
                        entity_key_serialization_version=config.entity_key_serialization_version,
                    )
                    if raw_key
                    else None,
                    values,
                )
            )
    return documents
