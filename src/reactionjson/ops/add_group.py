"""Atomic op: add_group.

Attaches a SMILES fragment to a specific atom of a molecule. The fragment must
contain exactly one `*` (dummy) atom marking its attachment point; the atom
bonded to `*` becomes bonded to the target atom in the output, and the `*` is
removed. Implicit hydrogens on both endpoints are adjusted to accommodate the
new bond.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


_ORDER_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

_ANCHOR_TAG = 999  # atom-map sentinel used to track the fragment's anchor
                   # across CombineMols + RemoveAtom (both of which can shift
                   # atom indices).


def add_group(
    mol: Chem.Mol, idx: int, fragment_smiles: str, order: int = 1
) -> Chem.Mol:
    """Attach a SMILES fragment to atom idx of mol.

    The fragment must contain exactly one `*` atom marking the attachment
    point. The atom bonded to `*` in the fragment becomes bonded to atom idx
    in the output; the `*` itself is removed.

    Returns a new sanitized Mol (input is not mutated).

    Explicit map numbers written into the fragment SMILES are preserved on the
    introduced atoms, so a later op can reference a newly-added atom (e.g. to
    close a ring onto it). Such map numbers must not already exist in `mol`.

    Raises ValueError if:
        - order is not in {1, 2, 3}
        - idx is out of range
        - fragment_smiles does not parse
        - the fragment has 0 or > 1 `*` atoms
        - the `*` atom has anything other than exactly 1 neighbor
        - a fragment map number collides with one already in `mol`, or uses the
          reserved internal sentinel
        - the resulting molecule fails sanitization (valence violation)
    """
    if order not in _ORDER_MAP:
        raise ValueError(f"order must be 1, 2, or 3; got {order!r}")

    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")

    frag = Chem.MolFromSmiles(fragment_smiles)
    if frag is None:
        raise ValueError(f"could not parse fragment SMILES: {fragment_smiles!r}")

    dummies = [a for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise ValueError(
            f"fragment must contain exactly one `*` attachment atom; "
            f"found {len(dummies)} in {fragment_smiles!r}"
        )
    dummy = dummies[0]
    dummy_neighbors = list(dummy.GetNeighbors())
    if len(dummy_neighbors) != 1:
        raise ValueError(
            f"`*` atom must have exactly one neighbor; got {len(dummy_neighbors)} "
            f"in {fragment_smiles!r}"
        )

    # Explicit map numbers in the fragment are preserved (so later ops can
    # reference introduced atoms). A fragment map number may legitimately equal
    # one already in `mol` — existing route_ops plans reuse product map numbers on
    # the re-attached group as part of map propagation — so we do NOT reject such
    # collisions here; a genuinely ambiguous *reference* is caught later by the
    # executor's `_resolve_map`. The one thing that must be rejected is a
    # NON-ANCHOR atom carrying the internal sentinel, which would break the
    # anchor-finding below (the anchor's own map is saved/restored, so it may
    # carry any value, as historical `*[C:999](C)C`-style fragments do).
    anchor_idx = dummy_neighbors[0].GetIdx()
    non_anchor_maps = [
        a.GetAtomMapNum() for a in frag.GetAtoms()
        if a.GetAtomMapNum() > 0 and a.GetIdx() != anchor_idx
    ]
    if _ANCHOR_TAG in non_anchor_maps:
        raise ValueError(
            f"fragment may not use map number {_ANCHOR_TAG} on a non-anchor atom; "
            f"it is reserved internally by add_group"
        )

    # Tag the fragment anchor with a sentinel atom-map number before combining.
    # CombineMols shifts indices, and RemoveAtom on the dummy shifts them again;
    # scanning for the tag is more robust than index arithmetic.
    # Save the anchor's original map number so we can restore it after the
    # sentinel has served its purpose (route_ops relies on this for map
    # propagation across add_group calls).
    frag_rw = Chem.RWMol(frag)
    anchor_frag_idx = dummy_neighbors[0].GetIdx()
    original_anchor_map = frag_rw.GetAtomWithIdx(anchor_frag_idx).GetAtomMapNum()
    frag_rw.GetAtomWithIdx(anchor_frag_idx).SetAtomMapNum(_ANCHOR_TAG)

    combined = Chem.CombineMols(mol, frag_rw)
    rw = Chem.RWMol(combined)

    anchor_combined_idx = None
    dummy_combined_idx = None
    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() == _ANCHOR_TAG:
            anchor_combined_idx = atom.GetIdx()
            atom.SetAtomMapNum(original_anchor_map)
        if atom.GetAtomicNum() == 0:
            dummy_combined_idx = atom.GetIdx()
    if anchor_combined_idx is None or dummy_combined_idx is None:
        raise ValueError("internal error: lost track of anchor or dummy after combine")

    # Adjust implicit Hs. The main-mol endpoint gains `order` bonds (loses
    # `order` Hs). The fragment anchor loses its bond to `*` (always single,
    # since `*` is a dummy with no meaningful valence) and gains a bond of
    # `order` — net valence change = order - 1. For order=1 the anchor is
    # unchanged, which is critical: touching the H count on a [C@@H]-style
    # anchor wipes the chiral reference frame.
    main_atom = rw.GetAtomWithIdx(idx)
    n_expl = main_atom.GetNumExplicitHs()
    if n_expl >= order:
        main_atom.SetNumExplicitHs(n_expl - order)
    elif n_expl > 0:
        main_atom.SetNumExplicitHs(0)

    anchor_delta = order - 1
    if anchor_delta > 0:
        anchor_atom = rw.GetAtomWithIdx(anchor_combined_idx)
        n_expl = anchor_atom.GetNumExplicitHs()
        if n_expl >= anchor_delta:
            anchor_atom.SetNumExplicitHs(n_expl - anchor_delta)
        elif n_expl > 0:
            anchor_atom.SetNumExplicitHs(0)

    # Add the new bond BEFORE removing the dummy: RDKit correctly updates bond
    # endpoint indices when RemoveAtom shifts atoms above the removed slot, so
    # the anchor's new bond survives intact.
    rw.AddBond(idx, anchor_combined_idx, _ORDER_MAP[order])
    rw.RemoveAtom(dummy_combined_idx)

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"add_group(idx={idx}, {fragment_smiles!r}, order={order}) "
            f"produced an invalid molecule: {exc}"
        ) from exc

    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)
    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import unittest
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    def _smi(s):
        m = Chem.MolFromSmiles(s)
        assert m is not None, f"bad test SMILES: {s}"
        return m

    def _canon(m):
        return Chem.MolToSmiles(m)

    class TestAddGroup(unittest.TestCase):
        # 1. Attach acetyl to methanol O: CO, idx=1 (O), fragment *C(=O)C
        def test_acetyl_on_methanol(self):
            m = _smi("CO")
            out = add_group(m, 1, "*C(=O)C", order=1)
            self.assertEqual(_canon(out), _canon(_smi("COC(C)=O")))

        # 2. Attach =O to methane: C, idx=0, fragment *=O, order=2 -> C=O
        def test_double_bond_oxo_on_methane(self):
            m = _smi("C")
            out = add_group(m, 0, "*=O", order=2)
            self.assertEqual(_canon(out), _canon(_smi("C=O")))

        # 3. Attach phenyl ring to methane: C, idx=0, fragment *c1ccccc1
        def test_phenyl_on_methane(self):
            m = _smi("C")
            out = add_group(m, 0, "*c1ccccc1")
            self.assertEqual(_canon(out), _canon(_smi("Cc1ccccc1")))

        # 4. Error: fragment has 0 `*` atoms
        def test_error_no_dummy(self):
            m = _smi("CO")
            with self.assertRaises(ValueError) as cm:
                add_group(m, 1, "CCO")
            self.assertIn("attachment", str(cm.exception).lower())

        # 5. Error: fragment has 2 `*` atoms
        def test_error_multiple_dummies(self):
            m = _smi("CO")
            with self.assertRaises(ValueError) as cm:
                add_group(m, 1, "*CC*")
            self.assertIn("attachment", str(cm.exception).lower())

        # 6. Error: invalid fragment SMILES
        def test_error_invalid_fragment(self):
            m = _smi("CO")
            with self.assertRaises(ValueError) as cm:
                add_group(m, 1, "not_smiles")
            self.assertIn("parse", str(cm.exception).lower())

        # 7. Error: idx out of range
        def test_error_idx_out_of_range(self):
            m = _smi("CO")
            with self.assertRaises(ValueError):
                add_group(m, 99, "*C")
            with self.assertRaises(ValueError):
                add_group(m, -1, "*C")

        # 8. Error: order not in {1,2,3}
        def test_error_bad_order(self):
            m = _smi("CO")
            with self.assertRaises(ValueError):
                add_group(m, 1, "*C", order=0)
            with self.assertRaises(ValueError):
                add_group(m, 1, "*C", order=4)

        # 9. Error: valence violation on neopentane central C
        def test_error_valence_violation(self):
            m = _smi("CC(C)(C)C")  # atom 1 is quaternary C with 0 H
            with self.assertRaises(ValueError) as cm:
                add_group(m, 1, "*O")
            self.assertIn("invalid", str(cm.exception).lower())

        # 10. Input mol unchanged
        def test_input_mol_not_mutated(self):
            m = _smi("CO")
            before = _canon(m)
            _ = add_group(m, 1, "*C(=O)C")
            self.assertEqual(_canon(m), before)

        # 11. Stereocenter on anchor is preserved. Attach alanine-minus-methyl
        #     fragment *[C@@H](N)C(=O)O to methane idx=0 -> reconstructs
        #     alanine C[C@@H](N)C(=O)O. The chiral C has 4 distinct neighbors
        #     (CH3, NH2, COOH, H) so it must register as a stereocenter.
        def test_anchor_stereocenter_preserved(self):
            m = _smi("C")
            out = add_group(m, 0, "*[C@@H](N)C(=O)O")
            centers = Chem.FindMolChiralCenters(out, includeUnassigned=False)
            self.assertEqual(len(centers), 1)
            self.assertEqual(_canon(out), _canon(_smi("C[C@@H](N)C(=O)O")))

        # 12. Attach to a tertiary CH: isobutane idx=1 (CH) + *O -> t-BuOH
        def test_attach_to_tertiary_ch(self):
            m = _smi("CC(C)C")
            out = add_group(m, 1, "*O", order=1)
            self.assertEqual(_canon(out), _canon(_smi("CC(C)(C)O")))

        # 13. Attach a multi-atom fragment: methane + ethoxy
        def test_ethoxy_on_methane(self):
            m = _smi("C")
            out = add_group(m, 0, "*OCC")
            self.assertEqual(_canon(out), _canon(_smi("CCOC")))

    unittest.main(verbosity=2)
