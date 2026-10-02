# reactionJson

Atomic graph edits for retrosynthesis, in both directions.

```python
from reactionjson import apply_ops, derive_ops, get_mapped_smiles

# apply a sequence of edits -> the precursors
ops = [{"op": "break_bond", "map_a": 3, "map_b": 4}]
smiles, err = apply_ops(ops, get_mapped_smiles("c1ccncc1"))

# a mapped reaction -> the ops that produce it
ops = derive_ops("<mapped_product>>><mapped_reactants>")
```

`derive_ops` and `apply_ops` are inverse: the ops derived from a reaction,
applied to its product, reproduce its reactants. That round trip is checked
before `derive_ops` returns.

## The vocabulary

Seven operations:

| op | params | effect |
|---|---|---|
| `break_bond` | `map_a`, `map_b` | removes one order: triple→double→single→severed |
| `add_bond` | `map_a`, `map_b`, `order?` | adds one order: none→single→double→triple |
| `add_group` | `map_idx`, `fragment_smiles` (exactly one `*`), `order?` | attaches a fragment |
| `set_explicit_h` | `map_idx`, `n` | sets the hydrogen count on one atom |
| `set_formal_charge` | `map_idx`, `charge` | sets the charge on one atom and refills its H to the new default valence |
| `set_stereocenter` | `map_idx`, `stereo` (R/S/null) | gives a centre an absolute CIP configuration, tagged or not; null clears it |
| `set_bond_stereo` | `map_a`, `map_b`, `stereo` (E/Z), `map_ref_a?`, `map_ref_b?` | sets alkene geometry |

`break_bond` and `add_bond` are a **ladder over bond order** — one order at a
time, and the only way to change a bond. Reducing a C=O to C–O is one
`break_bond`; severing it is two.

Atoms are addressed by **map number**, never by index, and map numbers are
resolved at the moment each op runs — so they stay valid after earlier ops add
or remove atoms.

## Kekulé handling

The molecule is kekulized **once**, before the first op, and stays that way for
the whole sequence. That matters: ops sanitize without re-perceiving
aromaticity, so the structure you were shown is the structure being edited, all
the way through. Re-kekulizing between ops could pick a different resonance
structure, and a bond you read as double could come back single.

Aromaticity is restored once at the end, so Kekulé form never leaks into the
returned SMILES.

## Errors

`apply_ops` never raises. It returns `(None, "Op [2] add_bond: ...")` naming the
op index that failed, which is what makes it usable in a retry loop.

## Install

```bash
uv pip install -e .
```
