# Milvus online store

## Description

The [Milvus](https://milvus.io/) online store provides support for materializing feature values into Milvus.

* The data model used to store feature values in Milvus is described in more detail [here](../../specs/online\_store\_format.md).

## Getting started
In order to use this online store, you'll need to install the Milvus extra (along with the dependency needed for the offline store of choice). E.g.

`pip install 'feast[milvus]'`

{% hint style="warning" %}
**Upgrading to milvus-lite 3.0.0+**

Feast supports both milvus-lite 2.x and 3.x. However, if you upgrade from milvus-lite 2.x.x to 3.0.0+, the `.db` files created by the original storage format are **not compatible** with the milvus-lite 3.0.0+ engine. You will need to re-import your data into a new database — automatic migration is not available.

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
| `embedding_dim` | `128` | Dimension of vector fields. |
| `index_type` | `FLAT` | Index type for vector fields with `vector_index=True`. |
| `metric_type` | `COSINE` | Default metric when a field does not set `vector_search_metric`. |
| `nlist` | `128` | `nlist` index parameter, used when `index_params` is unset. |
| `index_params` | unset | Index build parameters passed to Milvus, e.g. `{M: 16, efConstruction: 200}` for HNSW. Defaults to `{nlist: <nlist>}`, or no parameters for `AUTOINDEX`. |
| `search_params` | unset | Search parameters passed to Milvus, e.g. `{ef: 64}` for HNSW or `{level: 2}` for `AUTOINDEX`. Defaults to `{nprobe: 10}`, or no parameters for `AUTOINDEX`. |
| `consistency_level` | unset | `Strong`, `Bounded`, `Session` or `Eventually`. Sent with every read and search. When unset, Milvus uses the collection's level. |
| `collection_consistency_level` | unset | `Strong`, `Bounded`, `Session` or `Eventually`. Set when Feast creates a collection. When unset, Milvus uses its default (`Bounded`). |
| `partition_key` | unset | Field to use as the Milvus partition key in feature views that contain it. See [Partition key](#partition-key). |
| `vector_enabled` | `true` | Enables vector search. |
| `varchar_max_length` | `65535` | Default `max_length` of VARCHAR fields. Override per field with the `max_length` tag. |
| `enable_openai_compatible_store` | `false` | Store numeric features as native Milvus numeric types. |
| `storage_format` | `legacy` | `legacy` or `native`. See [Native storage format](#native-storage-format). |
| `num_partitions` | unset | Native only. Number of partitions for a new collection with a partition key. When unset, Milvus uses its default. |
| `partition_key_isolation` | unset | Native only. Enables partition-key isolation on a new collection with a partition key. When unset, Milvus uses its default. |
| `retry_mutations` | `true` | Native only. Set to `false` to disable SDK retries of upserts and deletes. See [Row deletion and mutation control](#row-deletion-and-mutation-control). |
| `mutation_timeout` | unset | Native only. SDK timeout in seconds for upserts and deletes. Must be unset when `retry_mutations` is `false`. |

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

Index parameters only apply when Feast creates a collection. To change them for an existing
collection, run `feast teardown` and `feast apply`, then materialize again.

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

The partition key field must be stored as `VARCHAR` or `INT64`. String fields are always `VARCHAR`,
and integer fields are `VARCHAR` unless native numeric types are enabled, in which case `Int64` works
and `Int32` does not.

{% hint style="warning" %}
The partition key only applies when Feast creates a collection. Existing collections are not
changed; Feast logs a warning for them. To add a partition key to an existing feature view, run
`feast teardown` and `feast apply`, then materialize again.
{% endhint %}

## Consistency level

By default Milvus uses `Bounded` consistency, so a read issued straight after materialization may
not see the newest writes for a short time. Set `consistency_level: Strong` if reads must always see
the latest writes, at the cost of higher read latency. See the
[Milvus consistency documentation](https://milvus.io/docs/consistency.md).

`consistency_level` applies to Feast's reads and searches and takes effect immediately.
`collection_consistency_level` sets the collection's own default, which Milvus uses for requests
that don't specify a level, such as those from other clients. It only applies when Feast creates a
collection; to change it, run `feast teardown` and `feast apply`, then materialize again.

## Collection loading

Feast creates collections together with their indexes, which makes Milvus load them straight away.
When Feast finds an existing collection it checks its load state and loads it only if needed.
Reads and searches never load collections, so a collection released outside Feast is only reloaded
the next time a Feast process first accesses it.

## Feature views without vectors

Milvus requires every collection to have a vector field. For feature views that have no vector
feature, Feast adds a 2-dimensional `_placeholder_vector` field with a FLAT index and fills it with zeros.
It is never returned or searched.

## Native storage format

Set `storage_format: native` to store each feature with a native Milvus type. The
default, `legacy`, does not change.

```yaml
online_store:
  type: milvus
  storage_format: native
  uri: http://localhost:19530
  index_type: AUTOINDEX
  partition_key: tenant_id
  num_partitions: 16
  consistency_level: Strong
```

The native format has these properties:

* Scalars, JSON and arrays keep their types. You can filter on them with Milvus expressions.
* Only fields with `vector_index=True` become dense vectors.
* Field tags declare BM25 functions, text analyzers and scalar indexes. Feast creates them with the collection.
* Feast validates an existing collection before it uses it. An incompatible collection causes an error. Feast does not change or drop it.
* `online_delete` deletes rows by entity key or by filter.

### Field types

| Feast field | Milvus field |
| --- | --- |
| String, Bool, Int32, Int64, Float32, Float64 | The same scalar type |
| Json, Struct, Map | JSON, including nested objects and arrays |
| Array of scalars | ARRAY with the same element type |
| Nested arrays, or arrays of structured values | JSON |
| Float array with `vector_index=True` | FLOAT_VECTOR. The dimension is `Field.vector_length`, or `embedding_dim` if that is unset |
| Bytes | Base64 VARCHAR. Feast reads decode it |
| Unix timestamp | INT64 seconds. Feast reads restore the timestamp |

Other Feast types, for example `ScalarMap`, `Set` and `Decimal`, cause a schema
error. A collection has one VARCHAR primary key: the serialized Feast entity key.
Each entity column is also stored, so you can filter on it.

Vectors must have the declared dimension and finite float32 values. A COSINE
vector must not be all zeros. A JSON value must not be a top-level string, must fit
in 65536 bytes, and its integers must fit in 64 bits. A collection can have at most
64 fields, including the key, the timestamps and the BM25 outputs. Feast rejects a
value that is not valid before it writes the batch. It does not truncate values.

When you push a DataFrame, encode nested `Json` values with `json.dumps(...)`.
Feast reads return the structured value.

### Field tags

All tag values are strings.

| Tag | Meaning |
| --- | --- |
| `max_length` | VARCHAR byte limit, also for strings in arrays. The default is `varchar_max_length`. |
| `milvus.max_capacity` | Maximum number of elements in an ARRAY, up to 4096. The default is 4096. |
| `milvus.nullable` | `true` or `false`. The default is `true`, but `false` for dense vectors. Entity and partition-key fields cannot be nullable. |
| `milvus.bm25` | Name of a sparse output field. Feast adds a BM25 function and a SPARSE_INVERTED_INDEX. |
| `milvus.analyzer_params` | JSON object with the Milvus analyzer parameters. |
| `milvus.multi_analyzer_params` | JSON object with `by_field` and `analyzers`. `analyzers` must have a `default` entry. Do not use it with `milvus.analyzer_params`. Requires PyMilvus 2.5.8 or newer. |
| `milvus.enable_match` | `true` enables TEXT_MATCH. Do not use it with `milvus.multi_analyzer_params`. |
| `milvus.index` | Index type, for example INVERTED, STL_SORT, NGRAM or HNSW. `false` disables a scalar index. |
| `milvus.index_params` | JSON object with the build parameters of that index. A vector field with `milvus.index` and no `milvus.index_params` gets no build parameters. |
| `milvus.json_indexes` | JSON list of typed paths, for example `[{"path":["region","price"],"cast_type":"double","index_type":"AUTOINDEX"}]`. |

This example adds two BM25 fields, a substring index and a sort index:

```python
Field(
    name="title",
    dtype=String,
    tags={
        "milvus.bm25": "title_sparse",
        "milvus.analyzer_params": '{"tokenizer":"standard","filter":["lowercase"]}',
    },
),
Field(name="description", dtype=String, tags={"milvus.bm25": "description_sparse"}),
Field(
    name="suggestion",
    dtype=String,
    tags={"milvus.index": "NGRAM", "milvus.index_params": '{"min_gram":2,"max_gram":3}'},
),
Field(name="popularity", dtype=Int32, tags={"milvus.index": "STL_SORT"}),
```

Milvus calculates the BM25 outputs. They are not Feast features and you do not
write them. A view with a BM25 field does not need the placeholder vector. To use
BM25 fields, search with the Milvus client. `retrieve_online_documents_v2` keeps
its LIKE text search.

### Schema validation

Feast compares an existing collection with the feature view before it uses it. It
checks field names, types, element types, nullability, the primary and partition
keys, dimensions, `max_length`, `max_capacity`, analyzers, TEXT_MATCH, BM25
functions, partition settings and the declared indexes. It does not compare field
descriptions. It accepts extra server parameters, such as mmap settings, and
indexes that an operator added. When the feature view changes, Feast validates
the collection again.

### Migrate to the native format

The native format cannot use a collection that the legacy format created. Feast
does not convert the data. To migrate without downtime:

1. Keep the current collection and its readers.
2. Create a feature view with a new name, or a new version if online feature view versioning is enabled.
3. Set `vector_index=True` on each embedding field. Set `vector_length` if it is different from `embedding_dim`.
4. Apply the new view with `storage_format: native`, then materialize it.
5. Make sure that reads and searches are correct. Then move the readers and writers to the new view.
6. Remove the old collection.

### Row deletion and mutation control

These methods are available only in the native format:

* `collection_name(config, view)` returns the collection name, with the version suffix if there is one.
* `ensure_schema(config, view)` creates a missing collection, or validates an existing one, and loads it.
* `prepare_write_batch(config, view, data)` validates and converts write tuples. It does no I/O and does not change `data`.
* `online_delete(config, view, entity_keys=[...])` deletes rows by entity key.
* `online_delete(config, view, filters=...)` deletes the rows that a `ComparisonFilter` or `CompoundFilter` selects. An empty filter is an error.

Give exactly one of `entity_keys` and `filters`. An empty key list does nothing.
Feast does not keep a tombstone, so a later write can create the row again. Within
one batch, the row with the latest event timestamp wins, then the latest created
timestamp, then the last row. Feast does not order writes across batches or
workers.

By default, PyMilvus retries upserts and deletes. `mutation_timeout` sets their
timeout. Set `retry_mutations: false` if your application handles uncertain
results itself. This disables all SDK retries, including retries after rate
limits, and requires `mutation_timeout` to be unset. A call without a timeout can
wait for a long time. After a transport error or a cancelled process, the result
on the server is unknown. Make sure that the earlier write is complete before you
send a conflicting write.

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
