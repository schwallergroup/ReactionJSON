"""reactionJson — atomic graph edits for retrosynthesis, both directions.

Two operations, inverse to each other:

    apply_ops(ops, mapped_smiles)   ops + a product  -> the precursors
    derive_ops(mapped_reaction)     a mapped reaction -> the ops that produce it

The op vocabulary is seven operations. ``break_bond`` and ``add_bond`` are a
ladder over bond order — they step one order at a time and are the only way to
change a bond, so reducing a C=O to C-O is one break_bond and severing it is two.

Structures are handled in Kekule form: the molecule is kekulized once, before
the first op, and stays that way for the whole sequence, so an aromatic ring is
editable like any other ring. Aromaticity is restored at the end, so Kekule form
never leaks into the result.

    >>> from reactionjson import apply_ops, derive_ops, get_mapped_smiles
    >>> ops = [{"op": "break_bond", "map_a": 3, "map_b": 4}]
    >>> apply_ops(ops, get_mapped_smiles("c1ccncc1"))
    ('[CH:1]1=[CH:2][CH2:3][NH:4][CH:5]=[CH:6]1', None)
"""

from reactionjson.executor import execute_ops as _execute_ops
from reactionjson.executor import get_mapped_smiles
from reactionjson.ops import VOCABULARY_V2 as OPS
from reactionjson.derive import OpDerivationError, derive_ops_for_reaction
from reactionjson.prompt import SYSTEM_PROMPT, user_prompt

__version__ = "0.1.0"
__all__ = [
    "apply_ops",
    "derive_ops",
    "get_mapped_smiles",
    "OPS",
    "OpDerivationError",
    "SYSTEM_PROMPT",
    "user_prompt",
]


def apply_ops(ops, mapped_smiles, *, kekulize=True):
    """Apply a sequence of edits to a mapped molecule.

    :param ops: list of op dicts, e.g. ``[{"op": "break_bond", "map_a": 3, "map_b": 4}]``
    :param mapped_smiles: the starting molecule, every atom carrying a map number
    :param kekulize: kekulize once before the first op (the default, and what the
        ladder semantics assume)
    :returns: ``(smiles, None)`` on success, ``(None, error_message)`` on failure.
        Never raises — a bad op comes back as an error string naming its index.
    """
    return _execute_ops(ops, mapped_smiles, vocabulary="v2", kekulize=kekulize)


def _fragments(smiles):
    """Multiset of unmapped canonical fragments, for comparing two sides."""
    from collections import Counter

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    out = []
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
        f = Chem.Mol(frag)
        for atom in f.GetAtoms():
            atom.SetAtomMapNum(0)
        out.append(Chem.MolToSmiles(f))
    return Counter(out)


def derive_ops(mapped_reaction, *, verify=True, **kwargs):
    """Derive the ops that turn a mapped product into its mapped reactants.

    :param mapped_reaction: ``"<mapped_product>>><mapped_reactants>"`` — the retro
        sense, atom-mapped on both sides
    :param verify: apply the derived ops and require that they reproduce the
        reactants. On by default: derivation can otherwise return ops that apply
        cleanly but give *different chemistry* (a bromide where the reaction had
        a hydroxyl), which is worse than failing. Measured on 400 recorded
        reactions, 25 came back wrong this way. Pass ``verify=False`` only if you
        are checking the result yourself.
    :returns: list of op dicts that :func:`apply_ops` turns back into those
        reactants.
    :raises OpDerivationError: when no op sequence is found, or when the derived
        one does not reproduce the reactants.
    """
    result = derive_ops_for_reaction(mapped_reaction, **kwargs)
    if getattr(result, "error", None):
        # derive.py reports a refusal on the result object rather than raising.
        # Dropping it turns "I cannot express this" into an empty op list, which
        # applies cleanly and changes nothing — a silent pass-through.
        raise OpDerivationError(result.error)
    ops = result.ops
    if not verify:
        return ops
    product, reactants = mapped_reaction.split(">>", 1)
    got, err = apply_ops(ops, product)
    if err is not None:
        raise OpDerivationError(f"derived ops do not apply: {err}")
    want, have = _fragments(reactants), _fragments(got)
    if want is None or have is None or want != have:
        raise OpDerivationError(
            "derived ops apply but give different products.\n"
            f"  expected: {sorted(want) if want else want}\n"
            f"  got:      {sorted(have) if have else have}"
        )
    return ops
