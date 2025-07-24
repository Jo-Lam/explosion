import itertools
import json
from collections import Counter, defaultdict

import pandas as pd
from dateutil.parser import parse


# Local Levenshtein implementation
def levenshtein(a: str, b: str) -> int:
    a, b = (a or ""), (b or "")
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            rem = curr[j-1] + 1
            sub = prev[j-1] + (ca != cb)
            curr.append(min(ins, rem, sub))
        prev = curr
    return prev[-1]

# Local Jaro–Winkler implementation
def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    if s1 == s2:
        return 1.0
    # … [same as before] …
    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    s1m = [False]*len1; s2m = [False]*len2; matches = 0
    for i in range(len1):
        for j in range(max(0, i-match_dist), min(i+match_dist+1, len2)):
            if not s2m[j] and s1[i]==s2[j]:
                s1m[i]=s2m[j]=True; matches+=1; break
    if matches == 0: return 0.0
    s1r=[s1[i] for i in range(len1) if s1m[i]]
    s2r=[s2[j] for j in range(len2) if s2m[j]]
    trans = sum(1 for a,b in zip(s1r,s2r) if a!=b)/2
    j = (matches/len1 + matches/len2 + (matches-trans)/matches)/3
    # common prefix
    prefix=0
    for a,b in zip(s1,s2):
        if a==b: prefix+=1
        else: break
    prefix=min(prefix,4)
    return j + prefix*p*(1-j)


def mismatch_positions(a: str, b: str) -> list[int]:
    return [i for i,(ca,cb) in enumerate(zip(a,b)) if ca!=cb]

def date_exploded(ref: str, var: str) -> bool:
    try:
        return parse(ref, dayfirst=True).date() != parse(var, dayfirst=True).date()
    except:
        return True

# build fake DataFrame
fields = {
    'id':         [1,1,1,  2,2,2],
    'first_name': ['john','jon','jahn','jhon','johm','johnny'],
    'last_name':  ['smith','smoth','smyth','smih','smiht','smith'],
    'sex':        ['M','Male','m','F','f','Female'],
    'dob':        ['1985-06-15','15-06-1985','1985/06/15',
                   '1985-06-16','1985-6-15','19850615'],
    'address':    ['123 Main St','123 Main Street','124 Main St',
                   '123 Main St.','123 Main St Apt1','123 Main St, Apt 1'],
    'postcode':   ['SW1A1AA','SW1A 1AA','SW1A1A','SW1A1AB','SW1A1BA','SW1A1AA'],
    'telephone':  ['02079460000','+442079460000','0207946000',
                   '2079460000','+44 20 79460000','0207946000'],
    'email':      ['john.smith@exmaple.com','john.smith@example.co.uk',
                   'johnsmith@example.com','john.smit@example.com',
                   'john.smith@example.org','john.smith@example.com']
}
df = pd.DataFrame(fields)
# lowercase
for c in df.columns:
    if c!='id':
        df[c]=df[c].str.lower()

# 1) build reference_by_id correctly:
reference_by_id = {
    fld: df.groupby('id')[fld]
           .agg(lambda s: s.iloc[-1])
           .to_dict()
    for fld in df.columns if fld!='id'
}

# —— 2) Build metadata —— #
metadata = {}
for fld, mapping in reference_by_id.items():
    metadata[fld] = {}
    # mapping is { id: reference_value }
    for rid, ref_val in mapping.items():
        # now sub is just the rows for this id
        sub = df[df['id'] == rid]
        # get variants (excluding the reference itself)
        variants = sub[fld][sub[fld] != ref_val]
        freq = Counter(variants)
        var_meta = {}
        for var, count in freq.items():
            info = {'frequency': count}
            if fld in ('first_name','last_name'):
                info.update({
                    'edit_distance':    levenshtein(ref_val, var),
                    'low_similarity':   jaro_winkler(ref_val, var) < 0.88,
                    'mismatch_first4':  any(p < 4 for p in mismatch_positions(ref_val, var))
                })
            elif fld in ('dob','postcode'):
                if fld == 'dob':
                    info['exploded'] = date_exploded(ref_val, var)
                else:
                    info['edit_distance'] = levenshtein(ref_val, var)
            var_meta[var] = info

        metadata[fld][ref_val] = {'variants': var_meta}

# write metadata.json
with open("metadata.json","w",encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print("✅ metadata.json written")


# 2) identify explosions per id
explosions_by_id = defaultdict(list)

for rid, group in df.groupby('id'):
    for fld, mapping in reference_by_id.items():
        ref_val = mapping[rid]
        variants = group[fld][group[fld] != ref_val].unique()

        exploded_any = False

        for var in variants:
            if fld in ('first_name','last_name'):
                explode = jaro_winkler(ref_val, var) < 0.88
            elif fld == 'dob':
                explode = date_exploded(ref_val, var)
            elif fld == 'postcode':
                explode = levenshtein(ref_val, var) > 0
            else:
                explode = True

            if explode:
                explosions_by_id[rid].append((fld, var))
                exploded_any = True

        # **fallback**: if no variant exploded for these key fields,
        # include the reference itself
        if fld in ('first_name', 'last_name', 'postcode') and not exploded_any:
            explosions_by_id[rid].append((fld, ref_val))


# 3) output
with open("explosions_by_id.json","w") as f:
    json.dump(explosions_by_id, f, indent=2)
print(json.dumps(explosions_by_id, indent=2))


# to create the exploded combinations:

records = []
for rec_id, variants in explosions_by_id.items():
    for field, variant in variants:
        records.append({"id": int(rec_id), "field": field, "variant": variant})

# Create DataFrame
df_long  = pd.DataFrame(records)
print(df_long)


# Build grouped exploded variants per id/field
grouped = (
    df_long
    .groupby(["id", "field"])["variant"]
    .unique()
    .unstack(fill_value=[])
)

# Define the full list of fields we care about (same as in reference_by_id)
all_fields = list(reference_by_id.keys())

# Generate all combinations per id, falling back to reference when needed
rows = []
for rec_id, row in grouped.iterrows():
    # For each field, if there are exploded variants use them;
    # otherwise use the single-element list [reference].
    variant_lists = []
    for fld in all_fields:
        exploded = list(row[fld]) if fld in row.index else []
        if exploded:
            variant_lists.append(exploded)
        else:
            # fallback to reference value
            variant_lists.append([ reference_by_id[fld][rec_id] ])
    
    # Cartesian product across these per-field lists
    for combo in itertools.product(*variant_lists):
        out = {"id": rec_id}
        out.update(dict(zip(all_fields, combo)))
        rows.append(out)

df_wide = pd.DataFrame(rows)

# reorder columns: id first, then alphabetical fields
df_wide = df_wide[["id"] + sorted(all_fields)]

# Save to Parquet
df_wide.to_parquet("exploded_combinations.parquet", index=False)
print(f"✅ Wrote {len(df_wide)} rows with all fields present to exploded_combinations.parquet")

print(df_wide)
