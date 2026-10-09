# Milvus online store

## Description

The [Milvus](https://milvus.io/) online store materializes each feature view into
one collection. Feature types determine the Milvus schema: nested metadata stays
JSON, ordinary arrays stay arrays, and fields marked for vector search become
vectors. This lets the same rows support Feast feature retrieval and Milvus
search with filters on their metadata.

Field tags can also declare BM25 functions, text analyzers and scalar indexes.
These are created with the collection when you apply the feature view. See
[native fields and search indexes](#native-fields-and-declarative-search-indexes)
and [upgrading an existing collection](#upgrading-an-existing-collection).

## Getting started
In order to use this online store, you'll need to install the Milvus extra (along with the dependency needed for the offline store of choice). E.g.

`pip install 'feast[milvus]'`

The Milvus extra requires PyMilvus 3.0.2 or newer and Milvus Lite 3.2.1 or newer.
The server integration tests use Milvus 3.0.2. Analyzer and index availability
also depends on the server version.

Use separate environments for the Milvus and Flink extras: their current
PyMilvus and Apache Beam dependencies require incompatible protobuf versions.

{% hint style="warning" %}
**Upgrading to milvus-lite 3.0.0+**

If you upgrade from milvus-lite 2.x, its `.db` files are **not compatible** with
the 3.x engine. Re-import your data into a new database; automatic file-format
migration is not available. This is separate from the collection-schema change
described in [upgrading an existing collection](#upgrading-an-existing-collection).

See the [milvus-lite GitHub page](https://github.com/milvus-io/milvus-lite) for more details.
{% endhint %}

You can get started by using any of the other templates (e.g. `feast init -t gcp` or `feast init -t snowflake` or `feast init -t aws`), and then swapping in Milvus as the online store as seen below in the examples.

## Examples

Using Milvus Lite, which stores data in a local file:

{% code title="feature_store.yaml" %}
```yaml
project: my_feature_repo
registry: data/registry.db
provider: local
online_store:
  type: milvus
  path: "data/online_store.db"
  embedding_dim: 128
  index_type: "FLAT"
  metric_type: "COSINE"
```
{% endcode %}

Connecting to a self-hosted Milvus server:

{% code title="feature_store.yaml" %}
```yaml
project: my_feature_repo
registry: data/registry.db
provider: local
online_store:
  type: milvus
  host: "http://localhost"
  port: 19530
  username: "username"
  password: "password"
  embedding_dim: 128
  index_type: "IVF_FLAT"
  metric_type: "COSINE"
```
{% endcode %}

Connecting to [Zilliz Cloud](https://zilliz.com/cloud) (managed Milvus) with an API key.
Read the token from an environment variable rather than committing it:

{% code title="feature_store.yaml" %}
```yaml
project: my_feature_repo
registry: data/registry.db
provider: local
online_store:
  type: milvus
  uri: "https://<your-cluster-endpoint>"   # Public Endpoint from the Zilliz Cloud console
  token: ${ZILLIZ_TOKEN}   # pragma: allowlist secret
  db_name: "default"
  embedding_dim: 768
  index_type: "AUTOINDEX"
  metric_type: "COSINE"
```
{% endcode %}

## Configuration options

| Option | Default | Description |
|:-------|:--------|:------------|
| `path` | `""` | Path to a Milvus Lite database file. Used when `provider: local` and `path` is set. |
| `host` | `http://localhost` | Milvus server host, including the scheme. |
| `port` | `19530` | Milvus server port. |
| `uri` | unset | Full endpoint, e.g. `https://<cluster>.zillizcloud.com:19530`. Takes precedence over `host`/`port`, and over `path`. |
| `username` / `password` | `""` | Credentials, sent as the token `username:password`. |
| `token` | unset | API key or `username:password`. Takes precedence over `username`/`password`. |
| `db_name` | unset | Milvus database to use. The database must already exist. Defaults to the server's `default` database. |
| `embedding_dim` | `128` | Fallback dimension for vector fields without `Field.vector_length`. |
| `index_type` | `FLAT` | Index type for vector fields with `vector_index=True`. |
| `metric_type` | `COSINE` | Default metric when a field does not set `vector_search_metric`. |
| `nlist` | `128` | `nlist` index parameter, used when `index_params` is unset. |
| `index_params` | unset | Index build parameters passed to Milvus, e.g. `{M: 16, efConstruction: 200}` for HNSW. Defaults to `{nlist: <nlist>}`, or no parameters for `AUTOINDEX`. |
| `search_params` | unset | Search parameters passed to Milvus, e.g. `{ef: 64}` for HNSW or `{level: 2}` for `AUTOINDEX`. Defaults to `{nprobe: 10}`, or no parameters for `AUTOINDEX`. |
| `consistency_level` | unset | `Strong`, `Bounded`, `Session` or `Eventually`. Sent with every read and search. When unset, Milvus uses the collection's level. |
| `collection_consistency_level` | unset | `Strong`, `Bounded`, `Session` or `Eventually`. Set when Feast creates a collection. When unset, Milvus uses its default (`Bounded`). |
| `partition_key` | unset | Field to use as the Milvus partition key in feature views that contain it. See [Partition key](#partition-key). |
| `num_partitions` | unset | Number of partitions when creating a collection with a partition key; unset uses the server default. |
| `partition_key_isolation` | unset | Whether to enable partition-key isolation when creating a partitioned collection; unset uses the server default. |
| `vector_enabled` | `true` | Enables vector search. |
| `varchar_max_length` | `65535` | Default `max_length` of VARCHAR fields. Override per field with the `max_length` tag. |
| `enable_openai_compatible_store` | `false` | Deprecated compatibility setting; numeric fields always use native types. |
| `retry_mutations` | `true` | Allow SDK retries of upserts and deletes. See [mutation control](#explicit-row-deletion-and-mutation-control). |
| `mutation_timeout` | unset | Optional SDK timeout for mutations; must be unset when `retry_mutations` is `false`. |

The full set of configuration options is available in [MilvusOnlineStoreConfig](https://rtd.feast.dev/en/latest/#feast.infra.online_stores.milvus.MilvusOnlineStoreConfig).

## Index and search parameters

`index_type`, `index_params` and `search_params` are passed through to Milvus, so any index type the
server supports can be used. On Zilliz Cloud, `AUTOINDEX` is recommended; tune the recall/latency
trade-off with the `level` search parameter:

```yaml
online_store:
  type: milvus
  index_type: "AUTOINDEX"
  search_params:
    level: 2
```

For HNSW:

```yaml
online_store:
  type: milvus
  index_type: "HNSW"
  index_params:
    M: 16
    efConstruction: 200
  search_params:
    ef: 64
```

Index parameters apply when Feast creates a collection. A different index definition
on an existing collection produces a schema drift error. Create a new feature view
or version and materialize it before switching readers; see [upgrading](#upgrading-an-existing-collection).

## Partition key

A [partition key](https://milvus.io/docs/use-partition-key.md) makes Milvus group rows by the key's
value, so searches filtered on it only scan the matching partitions. This suits multi-tenant data,
such as a catalogue shared by many brands.

Set the partition key per feature view with the `milvus.partition_key` tag:

```python
products = FeatureView(
    name="products",
    entities=[product],
    schema=[
        Field(name="product_id", dtype=Int64),
        Field(name="brand_id", dtype=String),
        Field(name="embedding", dtype=Array(Float32), vector_index=True),
        Field(name="title", dtype=String),
    ],
    source=products_source,
    tags={"milvus.partition_key": "brand_id"},
)
```

or for every feature view that has the field, with `partition_key: brand_id` in the online store
config. The tag takes precedence. Then filter on the key when retrieving:

```python
store.retrieve_online_documents_v2(
    features=["products:embedding", "products:title"],
    query=query_embedding,
    top_k=10,
    filters=ComparisonFilter(type="eq", key="brand_id", value="acme"),
)
```

The partition key field must be `String` or `Int64`, stored as `VARCHAR` or `INT64`.
`Int32` is not a valid Milvus partition-key type. Entity and partition-key fields
cannot be nullable.

{% hint style="warning" %}
Partition settings are checked against existing collections. A missing or different
partition key is an error; Feast does not change the collection. Use a new feature
view or version and materialize it before switching readers.
{% endhint %}

## Consistency level

By default Milvus uses `Bounded` consistency, so a read issued straight after materialization may
not see the newest writes for a short time. Set `consistency_level: Strong` if reads must always see
the latest writes, at the cost of higher read latency. See the
[Milvus consistency documentation](https://milvus.io/docs/consistency.md).

`consistency_level` applies to Feast's reads and searches and takes effect immediately.
`collection_consistency_level` sets the collection's own default, which Milvus uses for requests
that don't specify a level, such as those from other clients. It only applies when Feast creates a
collection; changing this configuration alone does not alter an existing collection.

## Collection loading

Feast creates collections together with their indexes. On first access, it checks
the schema and load state, and loads the collection if needed. Subsequent reads
and searches reuse the cached schema and do not repeat that load check. A
collection released outside Feast is checked again by a new store instance.

## Feature views without vectors

Milvus requires every collection to have a vector field. For feature views that have no vector
feature, Feast adds a 2-dimensional `_placeholder_vector` field with a FLAT index and fills it with zeros.
It is never returned or searched.

## Native fields and declarative search indexes

The standard `type: milvus` store uses native field types. No additional mode or
feature-view tag is needed. Text functions and indexes are declared on the fields
that need them. The same schema and value conversion are used for materialization,
online writes, feature retrieval and document retrieval.

```yaml
online_store:
  type: milvus
  uri: http://localhost:19530
  index_type: AUTOINDEX
  partition_key: tenant_id
  num_partitions: 16
  partition_key_isolation: false
  consistency_level: Strong
  collection_consistency_level: Strong
```

Each feature view still maps to one collection named `<project>_<view>` (with the
usual version suffix when enabled). Single and composite entity keys use Feast's
standard serialized primary key; each raw entity column is also stored for
filtering.

| Feast field | Native representation |
| --- | --- |
| String, Bool, Int32, Int64, Float32, Float64 | Corresponding scalar type |
| Json, Struct, Map | JSON, including nested objects and arrays |
| Array of scalar values | ARRAY with its scalar element type |
| Nested arrays or arrays of structured values | JSON |
| Float array with `vector_index=True` | FLOAT_VECTOR; dimension from `field.vector_length`, falling back to `embedding_dim` |
| Bytes | Base64 VARCHAR, decoded on Feast reads |
| Unix timestamp | INT64 seconds, restored on Feast reads |

Only fields explicitly marked `vector_index=True` become dense vectors. Vectors
must have finite float32-representable values and the declared dimension. A
COSINE vector must be nonzero. `field.description` is copied unchanged to the
Milvus field description, so applications may use it for their own metadata.
Unsupported Feast types fail schema validation. This includes `ScalarMap`: its
non-string keys cannot be preserved losslessly by Milvus JSON.
JSON string scalars, such as a feature whose entire JSON value is `"hello"`, are
rejected because the SDK treats strings as serialized JSON. Strings inside objects
and arrays remain supported. JSON
integers must fit the SDK's 64-bit range.

When pushing a DataFrame, JSON-encode nested `Json` values with `json.dumps(...)`
first. This preserves nested lists through Feast's DataFrame-to-protobuf
conversion. The Milvus store decodes the JSON input and writes a native JSON
value; normal Feast reads return the structured value.

Configure individual fields with string-valued tags:

| Tag | Meaning |
| --- | --- |
| `max_length` | VARCHAR byte limit; default `varchar_max_length` (array strings: 4096) |
| `max_capacity` | Scalar ARRAY capacity, up to 4096 |
| `milvus.nullable` | `true` (default for scalar/JSON/ARRAY) or `false`; entities, partition keys and dense vectors require `false` |
| `milvus.bm25` | Name of a derived sparse output field; creates a BM25 function and SPARSE_INVERTED_INDEX |
| `milvus.analyzer_params` | JSON object containing Milvus analyzer parameters |
| `milvus.multi_analyzer_params` | JSON object with `by_field` and `analyzers`, including a `default` analyzer; mutually exclusive with `analyzer_params` |
| `milvus.enable_match` | `true` enables TEXT_MATCH; cannot be combined with multiple analyzers |
| `milvus.index` | Scalar/vector index type, such as INVERTED, STL_SORT, NGRAM or HNSW; `false` disables a scalar index |
| `milvus.index_params` | JSON object containing that index's build parameters |
| `milvus.json_indexes` | JSON list of typed paths, e.g. `[{"path":["region","price"],"cast_type":"double","index_type":"AUTOINDEX"}]` |

For example, these fields create two BM25 search fields and a substring index:

```python
Field(
    name="title", dtype=String,
    tags={"milvus.bm25": "title_sparse",
          "milvus.analyzer_params": '{"tokenizer":"standard","filter":["lowercase","asciifolding"]}'},
),
Field(name="description", dtype=String,
      tags={"milvus.bm25": "description_sparse"}),
Field(name="suggestion", dtype=String,
      tags={"milvus.index": "NGRAM",
            "milvus.index_params": '{"min_gram":2,"max_gram":3}'}),
Field(name="popularity", dtype=Int32, tags={"milvus.index": "STL_SORT"}),
```

Derived sparse outputs are managed by Milvus and are not Feast features or write
inputs. A BM25-only view does not need a placeholder dense vector. Use the Milvus
client's search/hybrid-search APIs to select and combine these BM25 fields. Feast's
existing `retrieve_online_documents_v2` text-query behavior remains LIKE-based;
the declarative tags do not change its ranking behavior. Native JSON predicates
can address arbitrary keys; typed path indexes accelerate the declared paths.

Native schema validation checks field types, dimensions, descriptions, nullability,
analyzers, functions, partition settings and index definitions. Changes fail with
a drift error instead of silently reusing an incompatible collection. Collection
contracts are cached after validation; each new process validates the server.
Every JSON value must fit 65536 UTF-8 bytes and the complete schema must fit 64
fields, including the encoded key, timestamps and generated sparse outputs.
Oversized values fail before write; Feast does not truncate or split them.

Advanced analyzer and index options depend on the server and SDK versions. The
native integration suite is tested with Milvus 3.0.2 and PyMilvus 3.0.2. It requires
a disposable server; Milvus Lite does not cover all of these features:

```sh
export FEAST_TEST_MILVUS_URI=http://localhost:19530
python -m pytest --noconftest sdk/python/tests/integration/online_store/test_milvus_native_remote.py
```

## Upgrading an existing collection

This changes the default Milvus storage format. Previous versions stored most
non-vector values as VARCHAR, including serialized complex features, and treated
float arrays as vectors even without `vector_index=True`. The new format stores
typed scalars, JSON and ARRAY values, and only explicitly indexed float arrays
become vectors. `enable_openai_compatible_store` remains accepted for configuration
compatibility but no longer changes the schema.

Existing collections are checked before use. If their fields, functions, indexes
or partition settings differ, Feast raises an error with migration instructions.
It does not rewrite values, alter the schema or drop the collection automatically.
Even an empty collection with an old schema needs to be replaced.

To migrate without losing the existing serving data:

1. Keep the current reader and collection available during migration.
2. Create a feature view with a new name, or use a new version if online feature
   view versioning is enabled. Mark each embedding field with `vector_index=True`
   and set `vector_length` when it differs from `embedding_dim`.
3. Apply the new view and materialize its data, or replay complete rows through
   the normal online write path. The upgrade does not copy existing online data.
4. Check feature reads and search results, then switch readers and ongoing writers
   to the new view together.
5. Retire the old collection after its readers have moved.

Single and composite entity-key encoding, collection naming and feature-view
version suffixes follow the existing Feast conventions. Raw join-key columns are
also available for filtering; applications must still supply authorization filters.

## Explicit row deletion and mutation control

`MilvusOnlineStore.ensure_schema(config, registered_view)` creates a missing
collection, validates its schema and loads it once.
`collection_name(config, registered_view)` returns its standard name.
`prepare_write_batch(config, registered_view, data)` validates and converts normal
Feast write tuples without I/O or changing the input values.
`online_write_batch` uses the same validation before writing.

`online_delete(config, registered_view, entity_keys=[...])` deletes specific
serialized Feast keys. Alternatively, pass `filters=ComparisonFilter(...)` or a
`CompoundFilter` to delete matching rows, including all rows sharing one raw join
key in a composite-key view. Exactly one selector is required. An empty key list
is a no-op; an empty filter is rejected.

Deletion retains no tombstone: a later write can recreate the row. Timestamp
deduplication only applies within one batch. Cross-worker source ordering,
tombstones and atomic publication across views belong to the caller. Strong
consistency controls visibility and does not add compare-and-upsert semantics.

By default SDK mutation retries remain enabled. `mutation_timeout` optionally
sets their timeout. Publishers that coordinate ambiguous outcomes externally can
set `retry_mutations: false`; this requires `mutation_timeout: null` and disables
SDK retries, including rate-limit retries. PyMilvus 3.0.2 ignores the retry count
when a numeric timeout is supplied. A call without a timeout may wait indefinitely.
A transport error or process cancellation can leave an unknown server outcome;
the caller must establish that earlier writes have finished before safely issuing
a conflicting update.

## Functionality Matrix

The set of functionality supported by online stores is described in detail [here](overview.md#functionality).
Below is a matrix indicating which functionality is supported by the Milvus online store.

|                                                           | Milvus |
|:----------------------------------------------------------|:-------|
| write feature values to the online store                  | yes    |
| read feature values from the online store                 | yes    |
| update infrastructure (e.g. tables) in the online store   | yes    |
| teardown infrastructure (e.g. tables) in the online store | yes    |
| generate a plan of infrastructure changes                 | no     |
| support for on-demand transforms                          | yes    |
| readable by Python SDK                                    | yes    |
| readable by Java                                          | no     |
| readable by Go                                            | no     |
| support for entityless feature views                      | yes    |
| support for concurrent writing to the same key            | yes    |
| support for ttl (time to live) at retrieval               | yes    |
| support for deleting expired data                         | yes    |
| collocated by feature view                                | no     |
| collocated by feature service                             | no     |
| collocated by entity key                                  | no     |
| vector similarity search                                  | yes    |

To compare this set of functionality against other online stores, please see the full [functionality matrix](overview.md#functionality-matrix).
