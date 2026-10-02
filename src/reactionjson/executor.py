"""Deterministic executor for LLM-proposed atomic-op sequences.

Resolves map-number arguments to atom indices at runtime and applies each op
to the molecule state left by the previous op.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

from rdkit import Chem

from reactionjson.ops._sanitize import kekule_sequence
from reactionjson.ops import (
    add_bond,
    add_group,
    break_bond,
    change_atom,
    change_bond_order,
    clear_stereocenter,
    invert_stereocenter,
    remove_group,
    set_bond_stereo,
    set_explicit_h,
    set_formal_charge,
    set_stereocenter,
)


def ops_vocabulary() -> str:
    """The op set in force for this process: "v2" (nine ops) or "v1" (all twelve).

    One definition, read by every caller — the codegen policy that writes ops and
    the route_ops layer that replays them. They must agree: a v2 sequence that
    walks the bond-order ladder down an aromatic ring is rejected outright by a
    v1 replay, so a split setting turns a working search route into an
    unrenderable one.

    Replaying a historical v1 plan therefore means setting
    ``SYNTHELITE_OPS_VOCABULARY=v1`` for the whole process, not per call site.
    """
    return os.environ.get("SYNTHELITE_OPS_VOCABULARY", "v2")


def ops_kekulize() -> bool:
    """Whether structures are shown and edited in Kekule form. See :func:`ops_vocabulary`."""
    return os.environ.get("SYNTHELITE_OPS_KEKULIZE", "1") != "0"


def get_mapped_smiles(smiles: str) -> str:
    """Return SMILES where every atom carries map number = atom_idx + 1."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(mol)


def _resolve_map(mol: Chem.Mol, map_num: int) -> int:
    """Return the atom index for a given map number.

    Raises ValueError if no atom carries the map number, or if more than one
    does. The ambiguity check is important: atoms that ``add_group`` introduces
    without an explicit map number all share map 0, so a reference to map 0 (or
    any duplicated map number) would otherwise silently resolve to whichever
    atom happens to come first and corrupt the molecule.
    """
    matches = [atom.GetIdx() for atom in mol.GetAtoms()
               if atom.GetAtomMapNum() == map_num]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        surviving = sorted({a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0})
        raise ValueError(
            f"No atom with map number {map_num} in current molecule "
            f"(heavy atom count: {mol.GetNumAtoms()}). "
            f"Surviving map numbers: {surviving}"
        )
    raise ValueError(
        f"Ambiguous map number {map_num}: {len(matches)} atoms carry it. "
        f"Atoms introduced by add_group without an explicit map number all "
        f"share map 0; tag any atom you need to reference later with a unique, "
        f"unused map number in the fragment SMILES."
    )


def execute_ops(
    ops: list[dict],
    mapped_smiles: str,
    *,
    vocabulary: str = "v1",
    kekulize: bool = False,
) -> tuple[str | None, str | None]:
    """See :func:`_execute_ops`. This wrapper only scopes the kekule marker.

    While the marker is set, the ops sanitize without re-perceiving aromaticity,
    so the single kekulization below survives the whole sequence.
    """
    if not kekulize:
        return _execute_ops(ops, mapped_smiles, vocabulary=vocabulary, kekulize=False)
    with kekule_sequence():
        return _execute_ops(ops, mapped_smiles, vocabulary=vocabulary, kekulize=True)


def _execute_ops(
    ops: list[dict],
    mapped_smiles: str,
    *,
    vocabulary: str = "v1",
    kekulize: bool = False,
) -> tuple[str | None, str | None]:
    """Apply a sequence of atomic ops to the molecule encoded in mapped_smiles.

    Returns (result_smiles, error_message). Exactly one will be None.
    Map numbers in each op dict are resolved to atom indices at the time that
    op executes, so they remain valid even after prior ops remove or add atoms.
    Atom map numbers are always preserved in the returned SMILES. Atoms
    introduced by add_group come out with map=0 *unless* the fragment SMILES
    gives them an explicit map number, which is preserved so that later ops can
    reference the introduced atom (e.g. to close a ring onto it).

    :param vocabulary: "v1" (default) accepts every op, including the four kept
        only so historical plans, caches and rendered routes still replay. "v2"
        accepts the nine-op set the model is actually shown, and rejects the
        rest with a message naming the replacement. The default must stay "v1":
        139 of 141 recorded key steps and 95% of working recorded attempts use a
        v1-only op.
    :param kekulize: apply Chem.Kekulize(clearAromaticFlags=True) before the
        first op, so bond orders are explicit and the aromatic ring can be
        edited one order at a time. Once only: while the sequence runs, the ops
        sanitize with aromaticity perception off (atomic_ops/_sanitize.py), so
        the structure the model was shown survives to the last op.

        Must match what the prompt showed the model — if the prompt renders
        c1ccccc1 while this operates on C1=CC=CC=C1, every bond-order decision
        is made against the wrong structure. Aromaticity is
        re-perceived by the finalize sanitize, so Kekule form never leaks into
        the returned SMILES.
    """
    if vocabulary not in ("v1", "v2"):
        return None, f"unknown vocabulary {vocabulary!r}; expected 'v1' or 'v2'"

    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        return None, f"Could not parse mapped SMILES: {mapped_smiles!r}"

    if vocabulary == "v2":
        from reactionjson.ops import VOCABULARY_V2, VOCABULARY_V1_ONLY
        _REPLACEMENT = {
            "change_bond_order": "add_bond (delta>0) or break_bond (delta<0) — "
                                 "they now step the bond order one at a time",
            "remove_group": "break_bond — in the retro direction both fragments "
                            "are precursors, so nothing needs discarding",
            "change_atom": "not available: transmuting an element is not a "
                           "reaction. Disconnect and introduce the correct "
                           "fragment instead",
        }
        for i, op_dict in enumerate(ops):
            name = op_dict.get("op")
            if name in VOCABULARY_V1_ONLY:
                return None, (f"Op [{i}] {name}: not in vocabulary v2. "
                              f"Use {_REPLACEMENT[name]}.")
            if name not in VOCABULARY_V2:
                return None, f"Op [{i}]: unknown operation {name!r}"

    if kekulize:
        try:
            rw = Chem.RWMol(mol)
            Chem.Kekulize(rw, clearAromaticFlags=True)
            mol = rw.GetMol()
        except (Chem.KekulizeException, Chem.AtomKekulizeException, ValueError) as exc:
            return None, f"Kekulize: {type(exc).__name__}: {exc}"

    for i, op_dict in enumerate(ops):
        op = op_dict.get("op")
        try:
            if op == "break_bond":
                mol = break_bond(mol, _resolve_map(mol, op_dict["map_a"]),
                                      _resolve_map(mol, op_dict["map_b"]))
            elif op == "add_bond":
                mol = add_bond(mol, _resolve_map(mol, op_dict["map_a"]),
                                    _resolve_map(mol, op_dict["map_b"]),
                                    op_dict.get("order", 1))
            elif op == "change_bond_order":
                mol = change_bond_order(mol, _resolve_map(mol, op_dict["map_a"]),
                                             _resolve_map(mol, op_dict["map_b"]),
                                             op_dict["delta"])
            elif op == "change_atom":
                mol = change_atom(mol, _resolve_map(mol, op_dict["map_idx"]),
                                       op_dict["symbol"])
            elif op == "set_explicit_h":
                mol = set_explicit_h(mol, _resolve_map(mol, op_dict["map_idx"]),
                                          op_dict["n"])
            elif op == "add_group":
                mol = add_group(mol, _resolve_map(mol, op_dict["map_idx"]),
                                     op_dict["fragment_smiles"],
                                     op_dict.get("order", 1))
            elif op == "remove_group":
                mol = remove_group(mol, _resolve_map(mol, op_dict["map_idx"]),
                                        _resolve_map(mol, op_dict["map_root_idx"]))
            elif op == "invert_stereocenter":
                mol = invert_stereocenter(mol, _resolve_map(mol, op_dict["map_idx"]))
            elif op == "clear_stereocenter":
                mol = clear_stereocenter(mol, _resolve_map(mol, op_dict["map_idx"]))
            elif op == "set_formal_charge":
                mol = set_formal_charge(mol, _resolve_map(mol, op_dict["map_idx"]),
                                             op_dict["charge"])
            elif op == "set_stereocenter":
                mol = set_stereocenter(mol, _resolve_map(mol, op_dict["map_idx"]),
                                            op_dict["stereo"])
            elif op == "set_bond_stereo":
                ref_a = _resolve_map(mol, op_dict["map_ref_a"]) if op_dict.get("map_ref_a") else None
                ref_b = _resolve_map(mol, op_dict["map_ref_b"]) if op_dict.get("map_ref_b") else None
                mol = set_bond_stereo(mol, _resolve_map(mol, op_dict["map_a"]),
                                           _resolve_map(mol, op_dict["map_b"]),
                                           op_dict["stereo"], ref_a, ref_b)
            else:
                return None, f"Op [{i}]: unknown operation {op!r}"
        except (ValueError, KeyError, RuntimeError, Chem.AtomKekulizeException,
                Chem.KekulizeException, Chem.AtomValenceException) as exc:
            return None, f"Op [{i}] {op}: {type(exc).__name__}: {exc}\n  op_dict: {op_dict}"

    try:
        rw = Chem.RWMol(mol)
        clean = rw.GetMol()
        if kekulize:
            # The ops ran on a kekulized molecule with aromatic flags cleared, and
            # this block does NOT otherwise re-perceive aromaticity — it only
            # re-perceives stereo. Without this sanitize the returned SMILES comes
            # back as "[CH:1]1=[CH:2]..." instead of "[cH:1]1[cH:2]...", which is
            # the same molecule but a different string from every historical
            # artifact (tree JSONs, the reaction cache, rendered routes).
            Chem.SanitizeMol(clean)
        Chem.AssignStereochemistry(clean, cleanIt=True, force=True)
        result = Chem.MolToSmiles(clean)
    except (ValueError, RuntimeError, Chem.AtomKekulizeException,
            Chem.KekulizeException, Chem.AtomValenceException) as exc:
        return None, f"Finalize: {type(exc).__name__}: {exc}"
    if not result:
        return None, "Ops produced an empty molecule"
    return result, None
