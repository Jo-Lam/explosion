#!/usr/bin/env python3
"""
profile_exploded.py

Aggregate exploded identifier variants at scale:
- Phase 1: group & count unique (id, field, variant)
- Phase 2: join back to each id’s chosen reference (e.g. latest by timestamp)
- Phase 3: compute string metrics only on the compact variant set
- Phase 4: identify, per id, which (field,variant) combos are “explosions”
"""

import argparse
import json
from collections import Counter

import pandas as pd
from dateutil.parser import parse
from rapidfuzz.distance import JaroWinkler, Levenshtein


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_references(
    df: pd.DataFrame,
    id_col: str,
    ts_col: str,
    fields: list[str]
) -> pd.DataFrame:
    latest = (
        df
        .sort_values(ts_col)
        .groupby(id_col, as_index=False)
        .last()
    )
    return latest[[id_col] + fields]


def compute_frequencies(
    df: pd.DataFrame,
    id_col: str,
    fields: list[str]
) -> pd.DataFrame:
    pieces = []
    for fld in fields:
        grp = (
            df
            .groupby([id_col, fld], sort=False)
            .size()
            .reset_index(name='frequency')
            .assign(field=fld)
            .rename(columns={fld: 'variant'})
        )
        pieces.append(grp)
    return pd.concat(pieces, ignore_index=True)


def mismatch_positions(a: str, b: str) -> list[int]:
    return [i for i, (ca, cb) in enumerate(zip(a, b)) if ca != cb]


def date_exploded(ref: str, var: str) -> bool:
    """Treat as exploded if parse(ref) != parse(var)."""
    try:
        d1 = parse(ref, dayfirst=True).date()
        d2 = parse(var, dayfirst=True).date()
        return d1 != d2
    except:
        return True


def build_metadata(
    freq_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    id_col: str,
    fields: list[str]
) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for fld in fields:
        meta[fld] = {}
        f = freq_df[freq_df['field'] == fld]
        merged = f.merge(
            ref_df[[id_col, fld]].rename(columns={fld: 'reference'}),
            on=id_col,
            how='inner'
        )
        for ref_val, group in merged.groupby('reference', sort=False):
            subs = group[group['variant'] != ref_val]
            freq = Counter(subs['variant'])
            variants_meta: dict[str, dict] = {}
            for variant, count in freq.items():
                info = {'frequency': int(count)}
                if fld in ('first_name', 'last_name'):
                    ed  = Levenshtein.distance(ref_val, variant)
                    sim = JaroWinkler.similarity(ref_val, variant)
                    pos = mismatch_positions(ref_val, variant)
                    info.update({
                        'edit_distance':    ed,
                        'low_similarity':   sim < 0.88,
                        'mismatch_first4':  any(i < 4 for i in pos)
                    })
                elif fld in ('dob', 'postcode'):
                    # date: exploded if actual date differs
                    if fld == 'dob':
                        info['exploded'] = date_exploded(ref_val, variant)
                    else:
                        info['edit_distance'] = Levenshtein.distance(ref_val, variant)
                # other fields: only frequency
                variants_meta[variant] = info
            meta[fld][ref_val] = {'variants': variants_meta}
    return meta


def find_explosions(
    freq_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    id_col: str,
    fields: list[str]
) -> dict[int, list[tuple[str,str]]]:
    # melt references into (id, field, reference)
    refs = ref_df.melt(
        id_vars=[id_col],
        value_vars=fields,
        var_name='field',
        value_name='reference'
    )
    merged = freq_df.merge(refs, on=[id_col, 'field'], how='inner')

    def should_explode(row):
        fld = row['field']
        ref = row['reference']
        var = row['variant']
        if var == ref:
            return False
        if fld in ('first_name', 'last_name'):
            return JaroWinkler.similarity(ref, var) < 0.88
        if fld == 'dob':
            return date_exploded(ref, var)
        if fld == 'postcode':
            return Levenshtein.distance(ref, var) > 0
        # all other fields: include every variant
        return True

    merged['explode'] = merged.apply(should_explode, axis=1)

    explosions = {}
    for rid, group in merged.groupby(id_col):
        exploded = group[group['explode']]
        pairs = list(zip(exploded['field'], exploded['variant']))

        # Now ensure we include the reference itself if no variant exploded
        for fld in fields:
            refs_for_id = ref_df.loc[ref_df[id_col] == rid, fld].iloc[0]
            if fld in ('first_name', 'last_name', 'postcode'):
                # if no exploded variant in this field, add (field, reference)
                if not exploded[exploded['field'] == fld].any():
                    pairs.append((fld, refs_for_id))

        explosions[rid] = pairs

    return explosions


def main():
    p = argparse.ArgumentParser(description="Profile exploded identifiers at scale")
    p.add_argument("-i", "--input",  required=True, help="Path to exploded data (Parquet)")
    p.add_argument("-o", "--output", default="metadata.json", help="Path to write metadata JSON")
    p.add_argument("-e", "--explosions-output", default=None,
                   help="Path to write explosions-by-id JSON (optional)")
    p.add_argument("--id-col", default="id", help="Column name for record ID")
    p.add_argument("--ts-col", default="ts", help="Column name for timestamp to pick reference")
    p.add_argument("-f", "--fields", nargs="+",
                   default=["first_name","last_name","sex","dob",
                            "address","postcode","telephone","email"],
                   help="Identifier fields to profile")
    args = p.parse_args()

    df = load_data(args.input)
    # normalize case
    for fld in args.fields:
        df[fld] = df[fld].astype(str).str.lower()

    freq_df = compute_frequencies(df, args.id_col, args.fields)
    ref_df  = get_references(df, args.id_col, args.ts_col, args.fields)

    metadata = build_metadata(freq_df, ref_df, args.id_col, args.fields)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ metadata written to {args.output}")

    explosions = find_explosions(freq_df, ref_df, args.id_col, args.fields)
    if args.explosions_output:
        with open(args.explosions_output, "w", encoding="utf-8") as f:
            json.dump(explosions, f, indent=2)
        print(f"✅ explosions-by-id written to {args.explosions_output}")
    else:
        # print to stdout
        print(json.dumps(explosions, indent=2))


if __name__ == "__main__":
    main()
