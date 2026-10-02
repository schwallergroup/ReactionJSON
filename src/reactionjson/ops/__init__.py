"""Atomic molecular-editing operations for LLM-composed retrosynthesis.

Each op takes an RDKit Mol plus atom-level arguments, returns a new sanitized
Mol, and raises ValueError on any invalid input or invalid resulting molecule.
Input Mols are never mutated.
"""

from .add_bond import add_bond
from .break_bond import break_bond
from .change_atom import change_atom
from .set_explicit_h import set_explicit_h
from .invert_stereocenter import invert_stereocenter
from .clear_stereocenter import clear_stereocenter
from .change_bond_order import change_bond_order
from .set_bond_stereo import set_bond_stereo
from .set_formal_charge import set_formal_charge
from .set_stereocenter import set_stereocenter
from .add_group import add_group
from .remove_group import remove_group

__all__ = [
    "add_bond",
    "break_bond",
    "change_atom",
    "set_explicit_h",
    "invert_stereocenter",
    "clear_stereocenter",
    "change_bond_order",
    "set_bond_stereo",
    "set_formal_charge",
    "set_stereocenter",
    "add_group",
    "remove_group",
]


#: The op names the LLM is shown under vocabulary="v2". Nine primitives, with
#: break_bond/add_bond acting as a bond-order ladder that subsumes
#: change_bond_order. add_group is kept: it already preserves explicit map tags
#: in the fragment SMILES, so a later add_bond can reference the atoms it
#: introduced.
VOCABULARY_V2 = frozenset({
    "break_bond",
    "add_bond",
    "add_group",
    "set_explicit_h",
    "invert_stereocenter",
    "clear_stereocenter",
    "set_bond_stereo",
    "set_formal_charge",
    "set_stereocenter",
})

#: Retained only so historical plans, caches and rendered routes still replay:
#: 139 of 141 recorded key steps and 95% of working recorded attempts use at
#: least one of these. Never advertised to the model.
VOCABULARY_V1_ONLY = frozenset({
    "change_bond_order",   # subsumed by the add_bond/break_bond ladder
    "remove_group",        # discarded the detached fragment instead of keeping it
    "change_atom",         # element transmutation is not a reaction
})

VOCABULARY_V1 = VOCABULARY_V2 | VOCABULARY_V1_ONLY
