# KNN reference-library format

OsteoPilot does not distribute KNN reference data. To use
`find_similar_patients_by_block_hotelling_t2`, supply an external, writable
reference library through the required `reference_library` argument.

The library stores only per-lacuna morphometry and its metadata. Do not place
raw volumes, TIFF stacks, segmentation masks, evaluation results, or patient
exports in it.

## Required directory layout

```text
<reference_library>/
  <sample_identifier>/
    metadata.json
    morphometry.csv

  <another_sample_identifier>/
    metadata.json
    morphometry.csv
```

The folder name and `metadata.json.sample_identifier` must be identical. Every
sample folder is one reference block. A library must contain at least one valid
sample; a useful classification library needs enough samples for the requested
`top_k` and at least five valid morphometry rows per reference block.

The tool creates an optional `.knn_cache.json` next to each CSV after its first
use. It is a derived cache, not input data: it may be deleted at any time and
must not be versioned.

An optional `feature_policy.json` at the root changes the feature policy for
the entire library. If omitted, the built-in 22-feature policy below is used.

## `metadata.json`

Each sample requires this JSON object. `class_label` may be any non-empty
label, but all labels in one classification experiment should use a consistent
vocabulary.

```json
{
  "schema_version": 1,
  "sample_identifier": "sample_a_step0_Z0_z0000-0070_sectionT1",
  "patient_id": "sample_a",
  "class_label": "class_1",
  "step": 0,
  "z_index": 0,
  "slice_start": 0,
  "slice_end": 70,
  "section_name": "T1",
  "block_depth_px": 70,
  "morphometry_csv": "morphometry.csv"
}
```

Required fields and constraints:

| Field | Type | Constraint |
| --- | --- | --- |
| `sample_identifier` | string | Equal to the containing folder name. |
| `patient_id` | string | Non-empty stable identifier; used by `exclude_patient_id`. |
| `class_label` | string | Non-empty reference-class label. `disease` is accepted only as a legacy alias. |
| `step` | integer | Processing/acquisition step. |
| `z_index` | integer | Z-volume index. |
| `slice_start` | integer | First included slice. |
| `slice_end` | integer | Exclusive final slice; strictly greater than `slice_start`. |
| `morphometry_csv` | string | Relative path inside the same sample folder. Default: `morphometry.csv`. |

`section_name`, `block_depth_px`, `created_by`, and `source_note` are optional
but recommended. The tool can filter reference entries by any metadata field,
for example `{"section_name": "T2", "step": 0}`.

## `morphometry.csv`

This is a per-lacuna table: one row per valid lacuna, with a header row. It is
not a global summary, a density table, or a precomputed KNN-distance table.

The default policy requires these numeric columns:

```text
Centroid X
Centroid Y
Centroid Z
Lacuna BBox Size X (um)
Lacuna BBox Size Y (um)
Lacuna BBox Size Z (um)
Major Axis (um) (radius) (from PCA on volume)
Minor Axis (um) (radius) (from PCA on volume)
Minor to Major Axis Ratio
Surface Area (um^2)
Surface Area to Vol Ratio (raw)
Surface Area to Vol Ratio (um)
Volume (um^3)
Voxel Size (mm)
index Lc.Or1_x
index Lc.Or1_y
index Lc.Or1_z
index Lc.Or2_x
index Lc.Or2_y
index Lc.Or2_z
index Lc_Ob
index Lc_St
```

Additional columns are allowed and ignored by the default policy. The optional
`Block Depth (px)` column is checked when the active feature policy defines an
expected depth. Rows with missing or non-finite values in any active feature
column are excluded; at least five valid rows must remain.

The query CSV passed as `query_block_csv` follows the same column contract. It
does not need its own `metadata.json`, although keeping it next to one is useful
for traceability.

## Optional `feature_policy.json`

Use this only when all query and reference CSVs intentionally use a different
feature set. `feature_cols` replaces the default list and its order defines the
Hotelling T² vector order.

```json
{
  "feature_policy_id": "custom_policy_v1",
  "feature_cols": ["feature_a", "feature_b"],
  "shrinkage": 0.05,
  "expected_block_depth_px": 70
}
```

`shrinkage` must be numeric. Set `expected_block_depth_px` to `null` to disable
depth validation. Keep one policy per library; mixing feature policies within a
single library is unsupported.

## Invocation

```python
find_similar_patients_by_block_hotelling_t2(
    query_block_csv="data/dataset_a/query_block/morphometry.csv",
    reference_library="data/dataset_b/knn_reference_library",
    top_k=5,
    reference_filters={"section_name": "T2", "step": 0},
)
```

Reference data should remain outside the OsteoPilot repository, for example in
an access-controlled dataset location mounted under `data/dataset_b/`.
