"""Milvus schemas and native value conversion."""

from __future__ import annotations

import base64
import json
import math
import re
import struct
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pymilvus import CollectionSchema, DataType, Function, FunctionType, MilvusClient

from feast import FeatureView, Field
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.type_map import python_values_to_proto_values
from feast.types import Array, Float32, Float64
from feast.value_type import ValueType

if TYPE_CHECKING:
    from feast.infra.online_stores.milvus_online_store.milvus import (
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
                max_capacity=_limit(field, "max_capacity", 4096, 4096),
            )
            if spec["element_type"] == DataType.VARCHAR:
                spec["max_length"] = _limit(field, "max_length", 4096, 65535)
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
                if "milvus.index_params" not in field.tags:
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


def schema_contract(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize SDK/server representations for strict native schema comparison."""
    fields = []
    for field in schema["fields"]:
        params = dict(field.get("params", {}))
        for name in ("analyzer_params", "multi_analyzer_params"):
            if isinstance(params.get(name), str):
                params[name] = json.loads(params[name])
        for name in ("dim", "max_length", "max_capacity"):
            if name in params:
                params[name] = int(params[name])
        for name in ("enable_match", "enable_analyzer"):
            if name in params:
                params[name] = str(params[name]).lower() == "true"
        fields.append(
            {
                "name": field["name"],
                "type": int(field["type"]),
                "description": field.get("description", ""),
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
                "element_type": int(field.get("element_type", 0)),
            }
        )
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
        "fields": sorted(fields, key=lambda field: field["name"]),
        "functions": sorted(functions, key=lambda function: function["name"]),
        "auto_id": schema.get("auto_id", False),
        "enable_dynamic_field": schema.get("enable_dynamic_field", False),
    }


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
