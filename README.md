# Profile Exploded Identifiers

This repository contains **`profile_exploded.py`**, a high‑performance script for profiling and identifying "exploded" variants of key personal identifiers in large datasets.

## Overview

Large datasets often contain natural variations and historical values in fields such as names, dates, postcodes, and more. Understanding these variations helps:

* **Characterize** the distribution of real‑world data inconsistencies (metadata.json)
* **Drive deduplication** by focusing on truly distinct variants (explosions.json) - to minimise combinations of historical identifiers

## Context

- If we were to run full explosion, say in > 100M of records, we will be creating potentially x 10 times size of data.
- this script is a suggested solution to reduce this comparison pair & save computation.


## Files

* **`profile_explode.py`**: Main script with four phases:

  1. **Aggregate** unique `(id, field, variant)` frequencies
  2. **Select** each record's reference values (e.g., latest timestamp)
  3. **Compute** string metrics on the compact variant set (Levenshtein, Jaro–Winkler, date diff)
  4. **Identify** per‑ID "explosions": combinations of `(field, variant)` to include in dedupe

* **`README.md`**: This documentation.

## Requirements

* Python 3.8+
* pandas
* rapidfuzz
* python-dateutil

Install dependencies:

```bash
pip install pandas rapidfuzz python-dateutil pyarrow
```

## Usage

```bash
python profile_explode.py \
  --input exploded_data.parquet \
  --ts-col updated_at \
  --output metadata.json \
  --explosions-output explosions.json
```

* **`--input`**: Path to exploded Parquet dataset. Must include `id`, a timestamp column, and identifier fields.
* **`--ts-col`**: Column to use for selecting each record’s reference (default: `ts`).
* **`--fields`**: List of identifier columns to profile (default: first\_name, last\_name, sex, dob, address, postcode, telephone, email).

## Outputs

### `metadata.json`

A nested JSON structure detailing, for each field and reference value:

```json
{
  "first_name": {
    "jahn": {
      "variants": {
        "john": { "frequency": 10, "edit_distance": 1, "low_similarity": true, "mismatch_first4": true },
        "jon":  { ... }
      }
    },
    "johnny": { ... }
  },
  "dob": {
    "1985-06-15": {
      "variants": {
        "15-06-1985": { "frequency": 5, "exploded": false },
        "1985/06/15": { "frequency": 3, "exploded": false }
      }
    }
  },
  ...
}
```

**Purpose**:

* Understand natural variations and error distributions in your identifiers.
* Guide data quality assessments, schema design, and matching strategies.

### `explosions.json`

A mapping of each record **ID** to the list of `(field, variant)` pairs deemed worthy of deduplication:

```json
{
  "1": [["first_name","jon"], ["postcode","sw1a 1aa"], ["email","john.smith@example.co.uk"]],
  "2": [["last_name","smoth"], ["dob","1985-06-16"]],
  ...
}
```
**Inclusion Rules**:

* If no variants meet the explosion criteria, the reference value is included to ensure every field participates in the deduplication
* all other fields include every variant by default - see next steps 


**Purpose**:

* Direct input to our deduplication pipeline.
* Only explode on fields/values that represent big discrepancies not captured by string comparisons.

## Saving Exploded Variatns to Parquet

Convert json to dataframe, and write to parquet

```python
import json
import pandas as pd
import itertools

# 1) Load the explosions JSON
with open("explosions_by_id.json", "r") as f:
    explosions = json.load(f)

# 2) Build a long DataFrame
records = []
for rec_id, variants in explosions.items():
    for field, variant in variants:
        records.append({"id": int(rec_id), "field": field, "variant": variant})

df_long = pd.DataFrame(records)

# 3) Group by id and field
grouped = (
    df_long
    .groupby(["id", "field"])['variant']
    .unique()
    .unstack(fill_value=[])
)

# 4) Prepare all fields (matches your profile_explode.py --fields list)
all_fields = list(grouped.columns)

# 5) Generate all combinations per ID, fallback to reference if none exploded
#    You’ll need `reference_by_id` from your script in scope
rows = []
for rec_id, row in grouped.iterrows():
    for combo in itertools.product(*[row[f] if len(row[f])>0 else [reference_by_id[f][rec_id]] for f in all_fields]):
        out = {"id": rec_id}
        out.update(dict(zip(all_fields, combo)))
        rows.append(out)

df_wide = pd.DataFrame(rows)

# 6) Write to Parquet
output_path = "exploded_combinations.parquet"
df_wide.to_parquet(output_path, index=False)
print(f"✅ Wrote {len(df_wide)} rows to {output_path}")
```

## Next Steps

* **Adjust thresholds**: Modify the Jaro–Winkler cutoff or add additional rules in `profile_explode.py`.
* **Integrate** with Dedupe frameworks by feeding in `explosions.json`.
* **Extend** to additional identifier types or custom similarity metrics for different fields.

---

*Generated by profile\_explode.py tooling*
