from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import StrictStr, field_validator, model_validator
from pymilvus import (
    CollectionSchema,
    DataType,
    MilvusClient,
)
from pymilvus.client.types import LoadState

from feast import Entity
from feast.feature_view import (
    DUMMY_ENTITY_FIELD,
    DUMMY_ENTITY_ID,
    DUMMY_ENTITY_NAME,
    DUMMY_ENTITY_VAL,
    FeatureView,
)
from feast.filter_models import (
    ComparisonFilter,
    CompoundFilter,
    FilterTranslator,
    FilterType,
)
from feast.infra.infra_object import InfraObject
from feast.infra.key_encoding_utils import (
    deserialize_entity_key,
    serialize_entity_key,
)
from feast.infra.online_stores.helpers import compute_table_id
from feast.infra.online_stores.milvus_online_store import native
from feast.infra.online_stores.online_store import OnlineStore
from feast.infra.online_stores.vector_store import VectorStoreConfig
from feast.protos.feast.core.Registry_pb2 import Registry as RegistryProto
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import FeastConfigBaseModel, RepoConfig
from feast.type_map import (
    feast_value_type_to_python_type,
)
from feast.types import (
    PrimitiveFeastType,
    ValueType,
)

logger = logging.getLogger(__name__)


# Milvus requires every collection to have a vector field, so feature views
# without one get a small placeholder vector. Milvus servers reject vectors
# with fewer than 2 dimensions and non-finite values, and a collection can only
# be loaded once every vector field is indexed.
PLACEHOLDER_VECTOR_FIELD = "_placeholder_vector"
PLACEHOLDER_VECTOR_DIM = 2


def _milvus_escape_string(s: str) -> str:
    """Escape a string for safe use inside a Milvus single-quoted literal.

    Backslashes must be escaped first; otherwise a trailing backslash in the
    input would combine with the escaped quote to break out of the literal.
    Also escapes quotes and common control characters that could disrupt
    Milvus boolean expressions.
    """
    return (
        s.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _milvus_fmt(value: Any) -> str:
    """Format a Python value for use in a Milvus boolean expression.

    Handles numeric types natively (unquoted) so that Milvus performs
    numeric comparison instead of lexicographic string comparison.
    Bool is checked first because Python ``bool`` is a subclass of ``int``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{_milvus_escape_string(str(value))}'"


class MilvusFilterTranslator(FilterTranslator):
    """Translates Feast filters into Milvus boolean expression strings."""

    def translate(self, filters: FilterType) -> Optional[str]:
        if filters is None:
            return None
        return self._dispatch(filters)

    def translate_comparison(self, f: ComparisonFilter) -> str:
        key, value, op_type = f.key, f.value, f.type

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
            raise ValueError(
                f"Invalid filter key: {key!r}. Keys must be valid field identifiers."
            )

        milvus_ops = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}

        if op_type == "eq":
            return f"{key} == {_milvus_fmt(value)}"
        elif op_type == "ne":
            return f"{key} != {_milvus_fmt(value)}"
        elif op_type in milvus_ops:
            return f"{key} {milvus_ops[op_type]} {_milvus_fmt(value)}"
        elif op_type in ("in", "nin"):
            if not isinstance(value, list):
                raise ValueError(
                    f"'{op_type}' filter requires a list value, got {type(value)}"
                )
            formatted = [_milvus_fmt(v) for v in value]
            kw = "not in" if op_type == "nin" else "in"
            return f"{key} {kw} [{', '.join(formatted)}]"
        raise ValueError(f"Unsupported comparison operator: {op_type}")

    def translate_compound(self, f: CompoundFilter) -> str:
        if not f.filters:
            return ""
        clauses = []
        for sub_filter in f.filters:
            clause = self._dispatch(sub_filter)
            if clause:
                clauses.append(f"({clause})")
        if not clauses:
            return ""
        operator = " and " if f.type == "and" else " or "
        return operator.join(clauses)


class MilvusOnlineStoreConfig(FeastConfigBaseModel, VectorStoreConfig):
    """
    Configuration for the Milvus online store.
    NOTE: The class *must* end with the `OnlineStoreConfig` suffix.
    """

    type: Literal["milvus"] = "milvus"
    path: Optional[StrictStr] = ""
    host: Optional[StrictStr] = "http://localhost"
    port: Optional[int] = 19530
    # Full endpoint, e.g. a Zilliz Cloud https URI. Takes precedence over host/port.
    uri: Optional[StrictStr] = None
    # API key or "username:password". Takes precedence over username/password.
    token: Optional[StrictStr] = None
    # Milvus database to use. The database must already exist.
    db_name: Optional[StrictStr] = None
    index_type: Optional[str] = "FLAT"
    metric_type: Optional[str] = "COSINE"
    embedding_dim: Optional[int] = 128
    vector_enabled: Optional[bool] = True
    text_search_enabled: Optional[bool] = False
    nlist: Optional[int] = 128
    # Index build params for vector fields, e.g. {"M": 16, "efConstruction": 200}.
    # Defaults to {"nlist": nlist}, or {} for FLAT and AUTOINDEX.
    index_params: Optional[Dict[str, Any]] = None
    # Search params, e.g. {"ef": 64}, or {"level": 2} for AUTOINDEX.
    # Defaults to {"nprobe": 10}, or {} for AUTOINDEX.
    search_params: Optional[Dict[str, Any]] = None
    # Sent with every read and search. When unset, Milvus uses the
    # collection's level.
    consistency_level: Optional[
        Literal["Strong", "Bounded", "Session", "Eventually"]
    ] = None
    # Set when Feast creates a collection. When unset, Milvus uses its
    # default (Bounded).
    collection_consistency_level: Optional[
        Literal["Strong", "Bounded", "Session", "Eventually"]
    ] = None
    # Field to use as the Milvus partition key in feature views that contain it.
    # A feature view's "milvus.partition_key" tag takes precedence.
    partition_key: Optional[StrictStr] = None
    username: Optional[StrictStr] = ""
    password: Optional[StrictStr] = ""
    # Deprecated compatibility setting; numeric fields always use native types.
    enable_openai_compatible_store: Optional[bool] = False
    varchar_max_length: Optional[int] = 65535
    num_partitions: Optional[int] = None
    partition_key_isolation: Optional[bool] = None
    # PyMilvus 3.0.2 ignores retry_times when a numeric timeout is supplied.
    retry_mutations: bool = True
    mutation_timeout: Optional[float] = None

    @model_validator(mode="after")
    def validate_mutation_options(self) -> MilvusOnlineStoreConfig:
        if self.mutation_timeout is not None and (
            not math.isfinite(self.mutation_timeout) or self.mutation_timeout <= 0
        ):
            raise ValueError("mutation_timeout must be finite and positive")
        if not self.retry_mutations and self.mutation_timeout is not None:
            raise ValueError("retry_mutations=False requires mutation_timeout=None")
        if self.num_partitions is not None and not 1 <= self.num_partitions <= 4096:
            raise ValueError("num_partitions must be between 1 and 4096")
        return self

    @field_validator("varchar_max_length")
    @classmethod
    def validate_varchar_max_length(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 65535):
            raise ValueError(f"varchar_max_length must be between 1 and 65535, got {v}")
        return v


class MilvusOnlineStore(OnlineStore):
    """
    Milvus implementation of the online store interface.

    Attributes:
        _collections: Dictionary to cache Milvus collections.
    """

    def __init__(self) -> None:
        super().__init__()
        self.client: Optional[MilvusClient] = None
        self._collections: Dict[str, Any] = {}
        self._schema_contracts: Dict[str, str] = {}

    def _get_db_path(self, config: RepoConfig) -> str:
        assert (
            config.online_store.type == "milvus"
            or config.online_store.type.endswith("MilvusOnlineStore")
        )

        if config.repo_path and not Path(config.online_store.path).is_absolute():
            db_path = str(config.repo_path / config.online_store.path)
        else:
            db_path = config.online_store.path
        return db_path

    def _connect(self, config: RepoConfig) -> MilvusClient:
        if not self.client:
            online_config = config.online_store
            if (
                config.provider == "local"
                and online_config.path
                and not online_config.uri
            ):
                db_path = self._get_db_path(config)
                logger.info("Connecting to Milvus in local mode using %s", db_path)
                self.client = MilvusClient(db_path)
            else:
                uri = online_config.uri or f"{online_config.host}:{online_config.port}"
                logger.info("Connecting to Milvus remotely at %s", uri)
                client_kwargs: Dict[str, Any] = {
                    "uri": uri,
                    "token": _milvus_token(online_config),
                }
                if online_config.db_name:
                    client_kwargs["db_name"] = online_config.db_name
                self.client = MilvusClient(**client_kwargs)
        return self.client

    def collection_name(self, config: RepoConfig, table: FeatureView) -> str:
        """Return the standard project/view collection name, including versioning."""
        return _table_id(
            config.project, table, config.registry.enable_online_feature_view_versioning
        )

    def ensure_schema(self, config: RepoConfig, table: FeatureView) -> Dict[str, Any]:
        """Create a missing collection and load it; reject incompatible existing schemas."""
        return self._get_or_create_collection(config, table)

    def _build_schema(
        self, config: RepoConfig, table: FeatureView
    ) -> Tuple[CollectionSchema, Any]:
        table = _resolve_entityless_view(table)
        return native.build_schema(
            table,
            config.online_store,
            _get_composite_key_name(table),
            _partition_key_name(config.online_store, table),
            _index_build_params(config.online_store),
        )

    def _get_or_create_collection(
        self, config: RepoConfig, table: FeatureView
    ) -> Dict[str, Any]:
        schema, indexes = self._build_schema(config, table)
        contract = native.schema_contract(schema.to_dict())
        name = self.collection_name(config, table)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "schema": contract,
                    "indexes": [index.to_dict() for index in indexes],
                    "num_partitions": config.online_store.num_partitions,
                    "partition_key_isolation": config.online_store.partition_key_isolation,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if name in self._schema_contracts:
            if self._schema_contracts[name] != fingerprint:
                raise _schema_mismatch(name, "cached schema")
            return self._collections[name]
        client = self._connect(config)
        created = not client.has_collection(name)
        if created:
            options = _collection_consistency_kwargs(config.online_store)
            if _partition_key_name(config.online_store, table):
                if config.online_store.num_partitions is not None:
                    options["num_partitions"] = config.online_store.num_partitions
                if config.online_store.partition_key_isolation is not None:
                    options["properties"] = {
                        "partitionkey.isolation": str(
                            config.online_store.partition_key_isolation
                        ).lower()
                    }
            client.create_collection(
                collection_name=name, schema=schema, index_params=indexes, **options
            )
        description = client.describe_collection(name)
        if native.schema_contract(description) != contract:
            raise _schema_mismatch(name, "fields or functions")
        if _partition_key_name(config.online_store, table):
            if (
                config.online_store.num_partitions is not None
                and description.get("num_partitions")
                != config.online_store.num_partitions
            ):
                raise _schema_mismatch(name, "partition count")
            properties = description.get("properties", {})
            if isinstance(properties, list):
                properties = {entry["key"]: entry["value"] for entry in properties}
            actual_isolation = (
                str(properties.get("partitionkey.isolation", "false")).lower() == "true"
            )
            if (
                config.online_store.partition_key_isolation is not None
                and actual_isolation != config.online_store.partition_key_isolation
            ):
                raise _schema_mismatch(name, "partition isolation")
        actual_indexes = [
            client.describe_index(name, index_name)
            for index_name in client.list_indexes(name)
        ]
        if len(actual_indexes) != len(indexes):
            raise _schema_mismatch(name, "index set")
        for index in indexes:
            expected = index.to_dict()
            index_name = expected.pop("index_name") or expected["field_name"]
            # Lite names an index after its field, ignoring the supplied name.
            # Compare its field, type, metric and declared parameters instead.
            match = next(
                (
                    position
                    for position, actual in enumerate(actual_indexes)
                    if all(
                        str(actual.get(key)) == str(value)
                        for key, value in expected.items()
                    )
                ),
                None,
            )
            if match is None:
                raise _schema_mismatch(name, f"index {index_name}")
            actual_indexes.pop(match)
        if not created:
            self._ensure_loaded(name)
        self._collections[name] = description
        self._schema_contracts[name] = fingerprint
        return description

    def prepare_write_batch(
        self,
        config: RepoConfig,
        table: FeatureView,
        data: List[
            Tuple[EntityKeyProto, Dict[str, ValueProto], datetime, Optional[datetime]]
        ],
    ) -> List[Dict[str, Any]]:
        """Validate and convert a batch without I/O or changing inputs.

        This does not reserve rows or compare against previously stored timestamps.
        Only the newest row per key within this batch wins.
        """
        table = _resolve_entityless_view(table)
        schema, _ = self._build_schema(config, table)
        specs = {field.name: field.to_dict() for field in schema.fields}
        primary = _get_composite_key_name(table)
        entities = {field.name for field in table.entity_columns}
        fields = {field.name: field for field in table.schema}
        rows: Dict[str, Dict[str, Any]] = {}
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

        def micros(value: datetime) -> int:
            delta = value.replace(tzinfo=value.tzinfo or timezone.utc) - epoch
            return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds

        for entity_key, values, timestamp, created in data:
            entity_key = _normalize_entity_key(table, entity_key)
            if (
                len(entity_key.join_keys) != len(entity_key.entity_values)
                or set(entity_key.join_keys) != entities
                or len(entity_key.join_keys) != len(entities)
            ):
                raise ValueError(
                    "Entity key does not match the registered feature view join keys"
                )
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
                    raise ValueError(
                        f"Feature value disagrees with entity join key {name}"
                    )
                row[name] = decoded
            for name, field in fields.items():
                spec = specs[name]
                spec = {**spec.get("params", {}), **spec, "datatype": spec["type"]}
                row[name] = native.convert_value(field, row.get(name), to_storage=True)
                native.validate_value(
                    field,
                    spec,
                    row[name],
                    field.vector_search_metric
                    or config.online_store.metric_type
                    or "COSINE",
                )
            key = serialize_entity_key(
                entity_key,
                entity_key_serialization_version=config.entity_key_serialization_version,
            ).hex()
            if len(key) > (config.online_store.varchar_max_length or 65535):
                raise ValueError("Serialized entity key exceeds varchar_max_length")
            row.update(
                {
                    primary: key,
                    "event_ts": micros(timestamp),
                    "created_ts": micros(created) if created else 0,
                }
            )
            if native.PLACEHOLDER in specs:
                row[native.PLACEHOLDER] = [0.0, 0.0]
            if key not in rows or rows[key]["event_ts"] < row["event_ts"]:
                rows[key] = row
        return list(rows.values())

    def online_delete(
        self,
        config: RepoConfig,
        table: FeatureView,
        entity_keys: Optional[List[EntityKeyProto]] = None,
        *,
        filters: Optional[Union[ComparisonFilter, CompoundFilter]] = None,
    ) -> None:
        """Delete by entity keys or a typed filter, without retaining a tombstone.

        A later write can recreate a row. Cross-worker ordering and recovery from
        ambiguous mutations belong to the caller.
        """
        if (entity_keys is None) == (filters is None):
            raise ValueError("Provide exactly one of entity_keys or filters")
        if entity_keys == []:
            return
        table = _resolve_entityless_view(table)
        kwargs: Dict[str, Any]
        if entity_keys is not None:
            kwargs = {
                "ids": [
                    serialize_entity_key(
                        _normalize_entity_key(table, key),
                        entity_key_serialization_version=config.entity_key_serialization_version,
                    ).hex()
                    for key in entity_keys
                ]
            }
        else:
            expression = MilvusFilterTranslator().translate(filters)
            if not expression:
                raise ValueError("Deletion filter must not be empty")
            kwargs = {"filter": expression}
        collection = self.ensure_schema(config, table)
        self._connect(config).delete(
            collection_name=collection["collection_name"],
            **kwargs,
            **_mutation_kwargs(config.online_store),
        )

    def _ensure_loaded(self, collection_name: str) -> None:
        """Load an existing collection unless Milvus already has it loaded."""
        assert self.client is not None, "Milvus client is not initialized"
        load_state = self.client.get_load_state(collection_name).get("state")
        if load_state != LoadState.Loaded:
            self.client.load_collection(collection_name)

    def online_write_batch(
        self,
        config: RepoConfig,
        table: FeatureView,
        data: List[
            Tuple[
                EntityKeyProto,
                Dict[str, ValueProto],
                datetime,
                Optional[datetime],
            ]
        ],
        progress: Optional[Callable[[int], Any]],
    ) -> None:
        rows = self.prepare_write_batch(config, table, data)
        if not rows:
            return
        collection = self._get_or_create_collection(config, table)
        result = self._connect(config).upsert(
            collection_name=collection["collection_name"],
            data=rows,
            **_mutation_kwargs(config.online_store),
        )
        if result.get("upsert_count") != len(rows):
            raise RuntimeError("Milvus did not acknowledge every row in the batch")
        if progress:
            progress(len(data))

    def online_read(
        self,
        config: RepoConfig,
        table: FeatureView,
        entity_keys: List[EntityKeyProto],
        requested_features: Optional[List[str]] = None,
    ) -> List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]]:
        if not entity_keys:
            return []
        table = _resolve_entityless_view(table)
        collection = self.ensure_schema(config, table)
        fields = {field.name: field for field in table.schema}
        requested = (
            requested_features
            if requested_features is not None
            else [field.name for field in table.features]
        )
        if set(requested) - set(fields):
            raise ValueError(
                "Requested features are not in the native Milvus view schema"
            )
        primary = _get_composite_key_name(table)
        keys = [
            serialize_entity_key(
                _normalize_entity_key(table, key),
                entity_key_serialization_version=config.entity_key_serialization_version,
            ).hex()
            for key in entity_keys
        ]
        hits = self._connect(config).get(
            collection_name=collection["collection_name"],
            ids=keys,
            output_fields=[primary, "event_ts", *requested],
            **_consistency_kwargs(config.online_store),
        )
        by_key = {hit[primary]: hit for hit in hits}
        result: List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]] = []
        for key in keys:
            hit = by_key.get(key)
            if hit is None:
                result.append((None, None))
                continue
            values = {
                name: native.to_proto(fields[name], hit.get(name)) for name in requested
            }
            result.append(
                (datetime.fromtimestamp(hit["event_ts"] / 1e6, timezone.utc), values)
            )
        return result

    def update(
        self,
        config: RepoConfig,
        tables_to_delete: Sequence[FeatureView],
        tables_to_keep: Sequence[FeatureView],
        entities_to_delete: Sequence[Entity],
        entities_to_keep: Sequence[Entity],
        partial: bool,
    ) -> None:
        self.client = self._connect(config)
        for table in tables_to_keep:
            self._get_or_create_collection(config, table)

        # Always drop the base collection plus any "_v{N}" siblings, regardless of
        # the current versioning flag. This handles mixed-state repos where
        # versioning was toggled on/off across applies and would otherwise leave
        # orphan collections behind in Milvus.
        for table in tables_to_delete:
            self._drop_all_version_collections(config.project, table)

    def plan(
        self, config: RepoConfig, desired_registry_proto: RegistryProto
    ) -> List[InfraObject]:
        return []

    def teardown(
        self,
        config: RepoConfig,
        tables: Sequence[FeatureView],
        entities: Sequence[Entity],
    ) -> None:
        self.client = self._connect(config)
        # See update(): drop base + all "_v{N}" siblings to handle mixed-state repos.
        for table in tables:
            self._drop_all_version_collections(config.project, table)

    def retrieve_online_documents_v2(
        self,
        config: RepoConfig,
        table: FeatureView,
        requested_features: List[str],
        embedding: Optional[List[float]],
        top_k: int,
        distance_metric: Optional[str] = None,
        query_string: Optional[str] = None,
        filters: Optional[Union[ComparisonFilter, CompoundFilter]] = None,
        include_feature_view_version_metadata: bool = False,
    ) -> List[
        Tuple[
            Optional[datetime],
            Optional[EntityKeyProto],
            Optional[Dict[str, ValueProto]],
        ]
    ]:
        """
        Retrieve documents using vector similarity search or keyword search in Milvus.
        Args:
            config: Feast configuration object
            table: FeatureView object as the table to search
            requested_features: List of requested features to retrieve
            embedding: Query embedding to search for (optional)
            top_k: Number of items to return
            distance_metric: Distance metric to use (optional)
            query_string: The query string to search for using keyword search (optional)
        Returns:
            List of tuples containing the event timestamp, entity key, and feature values
        """
        table = _resolve_entityless_view(table)
        fields = {field.name: field for field in table.schema}
        self.client = self._connect(config)
        collection_name = _table_id(
            config.project, table, config.registry.enable_online_feature_view_versioning
        )
        collection = self._get_or_create_collection(config, table)
        if not config.online_store.vector_enabled:
            raise ValueError("Vector search is not enabled in the online store config")

        if embedding is None and query_string is None:
            raise ValueError("Either embedding or query_string must be provided")

        composite_key_name = _get_composite_key_name(table)

        output_fields = (
            [composite_key_name]
            + (requested_features if requested_features else [])
            + ["created_ts", "event_ts"]
        )
        assert all(
            field in [f["name"] for f in collection["fields"]]
            for field in output_fields
        ), (
            f"field(s) [{[field for field in output_fields if field not in [f['name'] for f in collection['fields']]]}] not found in collection schema"
        )

        # Find the vector search field if we need it
        ann_search_field = None
        if embedding is not None:
            for field in collection["fields"]:
                if (
                    field["type"] in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]
                    and field["name"] in output_fields
                ):
                    ann_search_field = field["name"]
                    break

        metadata_filter_expr = MilvusFilterTranslator().translate(filters)

        def _combine_exprs(*parts: Optional[str]) -> Optional[str]:
            """Combine non-empty Milvus boolean expressions with AND."""
            active = [p for p in parts if p]
            if not active:
                return None
            if len(active) == 1:
                return active[0]
            return " and ".join(f"({p})" for p in active)

        if (
            embedding is not None
            and query_string is not None
            and config.online_store.vector_enabled
        ):
            string_field_list = [
                f.name
                for f in table.features
                if isinstance(f.dtype, PrimitiveFeastType)
                and f.dtype.to_value_type() == ValueType.STRING
            ]

            if not string_field_list:
                raise ValueError(
                    "No string fields found in the feature view for text search in hybrid mode"
                )

            escaped_query = _milvus_escape_string(query_string)
            filter_expressions = []
            for field in string_field_list:
                if field in output_fields:
                    filter_expressions.append(f"{field} LIKE '%{escaped_query}%'")

            text_filter = " OR ".join(filter_expressions) if filter_expressions else ""
            combined_filter = _combine_exprs(text_filter, metadata_filter_expr)

            search_params = {
                "metric_type": distance_metric or config.online_store.metric_type,
                "params": _search_params(config.online_store),
            }

            results = self.client.search(
                collection_name=collection_name,
                data=[embedding],
                anns_field=ann_search_field,
                search_params=search_params,
                limit=top_k,
                output_fields=output_fields,
                filter=combined_filter,
                **_consistency_kwargs(config.online_store),
            )

        elif embedding is not None and config.online_store.vector_enabled:
            # Vector search only
            search_params = {
                "metric_type": distance_metric or config.online_store.metric_type,
                "params": _search_params(config.online_store),
            }

            results = self.client.search(
                collection_name=collection_name,
                data=[embedding],
                anns_field=ann_search_field,
                search_params=search_params,
                limit=top_k,
                output_fields=output_fields,
                filter=metadata_filter_expr,
                **_consistency_kwargs(config.online_store),
            )

        elif query_string is not None:
            string_field_list = [
                f.name
                for f in table.features
                if isinstance(f.dtype, PrimitiveFeastType)
                and f.dtype.to_value_type() == ValueType.STRING
            ]

            if not string_field_list:
                raise ValueError(
                    "No string fields found in the feature view for text search"
                )

            escaped_query = _milvus_escape_string(query_string)
            filter_expressions = []
            for field in string_field_list:
                if field in output_fields:
                    filter_expressions.append(f"{field} LIKE '%{escaped_query}%'")

            text_filter = " OR ".join(filter_expressions)

            if not text_filter:
                raise ValueError(
                    "No text fields found in requested features for search"
                )

            combined_filter = _combine_exprs(text_filter, metadata_filter_expr)

            query_results = self.client.query(
                collection_name=collection_name,
                filter=combined_filter or text_filter,
                output_fields=output_fields,
                limit=top_k,
                **_consistency_kwargs(config.online_store),
            )

            results = [
                [{"entity": entity, "distance": -1.0}] for entity in query_results
            ]
        else:
            raise ValueError(
                "Either vector_enabled must be True for embedding search or query_string must be provided for keyword search"
            )

        result_list = []
        for hits in results:
            for hit in hits:
                res = {}
                res_ts = None
                raw_key = hit.get("entity", {}).get(composite_key_name)
                entity_key_bytes = bytes.fromhex(raw_key) if raw_key else None
                entity_key_proto = (
                    deserialize_entity_key(entity_key_bytes)
                    if entity_key_bytes
                    else None
                )
                entity = hit.get("entity", {})
                event_ts = entity.get("event_ts")
                res_ts = (
                    datetime.fromtimestamp(event_ts / 1e6, timezone.utc)
                    if event_ts is not None
                    else None
                )
                res = {
                    name: native.to_proto(fields[name], entity.get(name))
                    for name in requested_features
                }
                raw_distance = hit.get("distance", None)
                res["distance"] = (
                    ValueProto(float_val=raw_distance)
                    if raw_distance is not None
                    else ValueProto()
                )
                result_list.append((res_ts, entity_key_proto, res if res else None))
        return result_list

    def _drop_all_version_collections(self, project: str, table: FeatureView) -> None:
        """Drop the base collection and every ``_v{N}`` versioned sibling.

        Mirrors the ``_drop_all_version_tables`` helpers in the MySQL/PostgreSQL
        online stores. Always called from ``update`` and ``teardown`` so a
        repo that toggles versioning on and off does not leave orphan
        collections behind in Milvus.
        """
        base = f"{project}_{table.name}"
        versioned_prefix = f"{base}_v"
        assert self.client is not None, "Milvus client is not initialized"
        for collection_name in self.client.list_collections():
            if collection_name == base or (
                collection_name.startswith(versioned_prefix)
                and collection_name[len(versioned_prefix) :].isdigit()
            ):
                self.client.drop_collection(collection_name)
                self._collections.pop(collection_name, None)
                self._schema_contracts.pop(collection_name, None)


def _schema_mismatch(collection: str, detail: str) -> ValueError:
    return ValueError(
        f"Milvus schema drift for collection {collection!r} ({detail}). "
        "The existing collection is incompatible with this feature view. "
        "Create a new feature view or feature-view version and rematerialize "
        "before switching readers. Feast will not alter or delete the existing collection."
    )


def _table_id(project: str, table: FeatureView, enable_versioning: bool = False) -> str:
    return compute_table_id(project, table, enable_versioning)


def _is_autoindex(online_config: MilvusOnlineStoreConfig) -> bool:
    return (online_config.index_type or "").upper() == "AUTOINDEX"


def _index_build_params(online_config: MilvusOnlineStoreConfig) -> Dict[str, Any]:
    """Build parameters; FLAT and AUTOINDEX need no tuning by default."""
    if online_config.index_params is not None:
        return dict(online_config.index_params)
    if (online_config.index_type or "FLAT").upper() in ("FLAT", "AUTOINDEX"):
        return {}
    return {"nlist": online_config.nlist}


def _search_params(online_config: MilvusOnlineStoreConfig) -> Dict[str, Any]:
    if online_config.search_params is not None:
        return dict(online_config.search_params)
    if _is_autoindex(online_config):
        return {}
    return {"nprobe": 10}


PARTITION_KEY_TAG = "milvus.partition_key"


def _partition_key_name(
    online_config: MilvusOnlineStoreConfig, table: FeatureView
) -> Optional[str]:
    """Return the partition key field for a feature view, if one is configured.

    The feature view's ``milvus.partition_key`` tag takes precedence over the
    store-level ``partition_key``, which only applies to feature views that
    contain that field.
    """
    tagged = table.tags.get(PARTITION_KEY_TAG)
    if tagged:
        return tagged
    configured = online_config.partition_key
    if configured and any(field.name == configured for field in table.schema):
        return configured
    return None


def _mutation_kwargs(online_config: MilvusOnlineStoreConfig) -> Dict[str, Any]:
    if not online_config.retry_mutations:
        return {"timeout": None, "retry_times": 0, "retry_on_rate_limit": False}
    if online_config.mutation_timeout is not None:
        return {"timeout": online_config.mutation_timeout}
    return {}


def _consistency_kwargs(online_config: MilvusOnlineStoreConfig) -> Dict[str, Any]:
    """Read and search kwargs; only pass a level when configured."""
    if online_config.consistency_level:
        return {"consistency_level": online_config.consistency_level}
    return {}


def _collection_consistency_kwargs(
    online_config: MilvusOnlineStoreConfig,
) -> Dict[str, Any]:
    """create_collection kwargs; only pass a level when configured."""
    if online_config.collection_consistency_level:
        return {"consistency_level": online_config.collection_consistency_level}
    return {}


def _milvus_token(online_config: MilvusOnlineStoreConfig) -> str:
    """Return the token to authenticate with: ``token``, else ``username:password``."""
    if online_config.token:
        return online_config.token
    if online_config.username and online_config.password:
        return f"{online_config.username}:{online_config.password}"
    return ""


def _get_composite_key_name(table: FeatureView) -> str:
    table = _resolve_entityless_view(table)
    return "_".join([field.name for field in table.entity_columns]) + "_pk"


def _resolve_entityless_view(table: FeatureView) -> FeatureView:
    """Feast hides its dummy entity on views passed to push/write ingestion."""
    if not table.entity_columns and (
        not table.entities or table.entities == [DUMMY_ENTITY_NAME]
    ):
        table = copy.copy(table)
        table.entities = [DUMMY_ENTITY_NAME]
        table.entity_columns = [DUMMY_ENTITY_FIELD]
    return table


def _normalize_entity_key(table: FeatureView, key: EntityKeyProto) -> EntityKeyProto:
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
