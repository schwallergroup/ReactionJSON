"""Atomic op: change_bond_order.

Changes the order of an existing single/double/triple bond by a signed delta.
Implicit hydrogens on the two endpoints are adjusted so that the new bond order
does not create a valence violation (or, on order decrease, so the atoms get
back the Hs they had freed).

Aromatic bonds are rejected: callers must kekulize first. To remove a bond
entirely (result order 0), callers must use `break_bond` instead.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


_ORDER_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def change_bond_order(mol: Chem.Mol, idx_a: int, idx_b: int, delta: int) -> Chem.Mol:
    """Change bond order between idx_a and idx_b by delta (+/- 1/2/3).

    Single -> Double is delta +1. Triple -> Single is delta -2. Etc.

    Returns a new sanitized Mol (input not mutated).

    Raises ValueError if:
        - delta is not in {-3, -2, -1, +1, +2, +3}
        - either index is out of range
        - no bond exists between the two atoms
        - the bond is aromatic (kekulize first)
        - the result would be order 0 (use break_bond) or > 3
        - the resulting molecule fails sanitization (valence violation)
    """
    if delta not in (-3, -2, -1, 1, 2, 3):
        raise ValueError(
            f"delta must be in {{-3,-2,-1,+1,+2,+3}}; got {delta!r}"
        )

    n = mol.GetNumAtoms()
    if not (0 <= idx_a < n) or not (0 <= idx_b < n):
        raise ValueError(
            f"atom index out of range: idx_a={idx_a}, idx_b={idx_b}, num_atoms={n}"
        )

    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None:
        raise ValueError(f"no bond exists between atoms {idx_a} and {idx_b}")

    # Reject aromatic bonds: GetBondTypeAsDouble() returns 1.5, which would
    # silently int-cast to 1. Check the BondType directly.
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        raise ValueError("cannot change order of aromatic bond; kekulize first")

    current = int(bond.GetBondTypeAsDouble())
    if current not in _ORDER_MAP:
        raise ValueError(
            f"bond between {idx_a} and {idx_b} has unsupported type "
            f"{bond.GetBondType()}; only SINGLE/DOUBLE/TRIPLE are supported"
        )

    new_order = current + delta
    if new_order <= 0:
        raise ValueError(
            f"delta={delta} would reduce bond order from {current} to "
            f"{new_order}; use break_bond to remove the bond"
        )
    if new_order > 3:
        raise ValueError(
            f"delta={delta} would raise bond order from {current} to "
            f"{new_order}; maximum supported order is 3"
        )

    rw = Chem.RWMol(mol)

    # Adjust implicit-H counts on both endpoints. When increasing bond order,
    # each endpoint loses |delta| Hs; when decreasing, each gains |delta| Hs.
    # RDKit's auto-adjust on SetBondType is unreliable when atoms carry explicit
    # H counts or have noImplicit set, so we handle both paths manually.
    for idx in (idx_a, idx_b):
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetNumExplicitHs() > 0 or atom.GetNoImplicit():
            new_h = atom.GetNumExplicitHs() - delta
            if new_h < 0:
                new_h = 0
            atom.SetNumExplicitHs(new_h)
            # Keep noImplicit as-is; caller-provided state is preserved.
        else:
            # Implicit-H atom: let sanitize recompute from scratch.
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)

    new_bond = rw.GetBondBetweenAtoms(idx_a, idx_b)
    new_bond.SetBondType(_ORDER_MAP[new_order])
    # Clear E/Z on the edited bond AND the `/`/`\` direction markers on all
    # bonds adjacent to either endpoint: AssignStereochemistry re-perceives
    # double-bond geometry from those directional singles, so leaving them in
    # place resurrects stale E/Z after a reduce+raise round-trip.
    new_bond.SetStereo(Chem.BondStereo.STEREONONE)
    new_bond.SetBondDir(Chem.BondDir.NONE)
    for end_idx in (idx_a, idx_b):
        for nbr_bond in rw.GetAtomWithIdx(end_idx).GetBonds():
            nbr_bond.SetBondDir(Chem.BondDir.NONE)

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"change_bond_order({idx_a}, {idx_b}, delta={delta}) produced an "
            f"invalid molecule: {exc}"
        ) from exc

    # Changing a bond order may create/destroy a stereocenter (e.g. C=C -> C-C
    # clears any E/Z that was on this bond, or a new sp2 center appears).
    # SanitizeMol does not re-perceive stereo, so force a fresh pass.
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

    class TestChangeBondOrder(unittest.TestCase):
        # 1. Single -> double: CC -> C=C
        def test_single_to_double(self):
            m = _smi("CC")
            out = change_bond_order(m, 0, 1, +1)
            self.assertEqual(_canon(out), _canon(_smi("C=C")))

        # 2. Double -> single: C=C -> CC
        def test_double_to_single(self):
            m = _smi("C=C")
            out = change_bond_order(m, 0, 1, -1)
            self.assertEqual(_canon(out), _canon(_smi("CC")))

        # 3. Single -> triple: CC -> C#C
        def test_single_to_triple(self):
            m = _smi("CC")
            out = change_bond_order(m, 0, 1, +2)
            self.assertEqual(_canon(out), _canon(_smi("C#C")))

        # 4. Triple -> double: C#C -> C=C
        def test_triple_to_double(self):
            m = _smi("C#C")
            out = change_bond_order(m, 0, 1, -1)
            self.assertEqual(_canon(out), _canon(_smi("C=C")))

        # Bonus: triple -> single.
        def test_triple_to_single(self):
            m = _smi("C#C")
            out = change_bond_order(m, 0, 1, -2)
            self.assertEqual(_canon(out), _canon(_smi("CC")))

        # 5. Error: bond does not exist.
        def test_error_bond_missing(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                change_bond_order(m, 0, 5, +1)
            # out-of-range should be caught first
            self.assertIn("out of range", str(cm.exception).lower())

        def test_error_bond_missing_in_range(self):
            m = _smi("CCC")  # atoms 0 and 2 exist but are not bonded
            with self.assertRaises(ValueError) as cm:
                change_bond_order(m, 0, 2, +1)
            self.assertIn("no bond", str(cm.exception).lower())

        # 6. Error: delta = 0.
        def test_error_delta_zero(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                change_bond_order(m, 0, 1, 0)
            self.assertIn("delta", str(cm.exception).lower())

        # 7. Error: order drops to 0 -> must point to break_bond.
        def test_error_order_zero_points_to_break_bond(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                change_bond_order(m, 0, 1, -1)
            self.assertIn("break_bond", str(cm.exception))

        # 8. Error: order would exceed 3.
        def test_error_order_exceeds_three(self):
            m = _smi("C=C")
            with self.assertRaises(ValueError) as cm:
                change_bond_order(m, 0, 1, +2)
            self.assertIn("maximum", str(cm.exception).lower())

        def test_error_order_exceeds_three_from_triple(self):
            m = _smi("C#C")
            with self.assertRaises(ValueError):
                change_bond_order(m, 0, 1, +1)

        # 9. Error: bond is aromatic.
        def test_error_aromatic_bond(self):
            m = _smi("c1ccccc1")
            with self.assertRaises(ValueError) as cm:
                change_bond_order(m, 0, 1, +1)
            self.assertIn("aromatic", str(cm.exception).lower())

        # 10. Error: valence violation. Neopentane "CC(C)(C)C": atom 1 is a
        #     quaternary C with 4 heavy neighbors and 0 Hs to give up. Bumping
        #     bond 0-1 to double would require valence 5 on atom 1 with no H
        #     pool to draw from, so sanitize must reject.
        def test_error_valence_violation(self):
            m = _smi("CC(C)(C)C")  # atoms: 0=CH3, 1=C (4 heavy nbrs, 0 H)
            self.assertEqual(m.GetAtomWithIdx(1).GetTotalNumHs(), 0)
            with self.assertRaises(ValueError) as cm:
                change_bond_order(m, 0, 1, +1)
            self.assertIn("invalid", str(cm.exception).lower())

        # 11. Input mol unchanged.
        def test_input_mol_not_mutated(self):
            m = _smi("CC")
            before = _canon(m)
            before_bond_type = m.GetBondBetweenAtoms(0, 1).GetBondType()
            _ = change_bond_order(m, 0, 1, +1)
            self.assertEqual(_canon(m), before)
            self.assertEqual(
                m.GetBondBetweenAtoms(0, 1).GetBondType(), before_bond_type
            )

        # 12. Round-trip: +1 then -1 recovers canonical SMILES.
        def test_round_trip_plus_minus(self):
            m = _smi("CC")
            up = change_bond_order(m, 0, 1, +1)
            down = change_bond_order(up, 0, 1, -1)
            self.assertEqual(_canon(down), _canon(m))

        def test_round_trip_plus2_minus2(self):
            m = _smi("CC")
            up = change_bond_order(m, 0, 1, +2)
            down = change_bond_order(up, 0, 1, -2)
            self.assertEqual(_canon(down), _canon(m))

        # 13. Stereo: C=C with E/Z, reduced to C-C, then put back -> any E/Z
        #     that was on the original double bond is gone (butane has no
        #     stereo). Tested indirectly via canonical SMILES.
        def test_stereo_cleared_on_reduce_and_reraise(self):
            # trans-2-butene: C/C=C/C
            m = _smi("C/C=C/C")
            # Reducing the double bond to single: result is n-butane (no stereo)
            reduced = change_bond_order(m, 1, 2, -1)
            self.assertEqual(_canon(reduced), _canon(_smi("CCCC")))
            # Raising it back up: the E-designation is gone, just "CC=CC".
            raised = change_bond_order(reduced, 1, 2, +1)
            self.assertEqual(_canon(raised), _canon(_smi("CC=CC")))
            # And the canonical SMILES of raised is NOT the E-form.
            self.assertNotEqual(_canon(raised), _canon(m))

    unittest.main(verbosity=2)
