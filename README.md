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
time. Reducing the pi bond in C=O to C–O is one `break_bond', breaking the sigma bond it is two.

## Asking an LLM for a step

`SYSTEM_PROMPT` explains the vocabulary to a model, and `user_prompt` renders the
product the way the executor will edit it (Kekulé, mapped) next to your strategy.
The model answers with JSON whose `"operations"` go straight into `apply_ops`:

```python
import json
from reactionjson import SYSTEM_PROMPT, apply_ops, get_mapped_smiles, user_prompt

mapped = get_mapped_smiles("CC(=O)OCC")
reply = your_llm(system=SYSTEM_PROMPT,
                 user=user_prompt(mapped, "retro-esterification: cut the ester C-O"))
precursors, err = apply_ops(json.loads(reply)["operations"], mapped)
```

Give `apply_ops` the same mapped SMILES you gave `user_prompt`, so the bond orders
the model read are the ones it edits. On failure `err` names the op that broke —
append it to the next user message and ask again. The reply also carries
`"analysis"` and a one-sentence `"reaction_conditions"`.

## Install

```bash
uv pip install -e .
```
