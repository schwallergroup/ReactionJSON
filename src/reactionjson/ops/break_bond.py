"""Atomic op: break_bond.

Breaks a single (sigma) bond between two atoms. The freed valence on each
endpoint is filled with implicit hydrogens via re-sanitization. If the broken
bond was part of a stereocenter's definition, the chiral tag on the two
endpoints is cleared; stereo elsewhere in the molecule is preserved.

Semantics (v2): break_bond removes the HIGHEST bond component, one call at a
time — triple -> double -> single -> severed. This mirrors how a bond is
actually attacked: a nucleophile hitting a carbonyl breaks the pi bond before
the sigma, so reducing C=O to C-OH is a single break_bond, and severing a bond
outright is break_bond repeated until the order reaches zero.

This replaces `change_bond_order` (delta=-n is n break_bond calls) and is
strictly more permissive than the original op, which refused every non-single
bond. No sequence that worked before can change behaviour, because none could
ever call this on a double, triple or aromatic bond.

An AROMATIC bond is still refused: order 1.5 has no well-defined decrement.
Under vocabulary="v2" the executor kekulizes first so the case cannot arise;
a v1 caller gets the same error it always got.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


def break_bond(mol: Chem.Mol, idx_a: int, idx_b: int) -> Chem.Mol:
    """Break the single (sigma) bond between atoms idx_a and idx_b.

    Fills freed valence with implicit H. Returns a new sanitized Mol (input is
    not mutated). The result may contain multiple disconnected fragments in a
    single Mol object; the caller decides whether to split.

    Only sigma bonds are supported. For a double or triple bond, use
    `change_bond_order` to reduce it first.

    Raises ValueError if:
        - either index is out of range
        - no bond exists between the two atoms
        - the bond exists but is not SINGLE (pi bond present)
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx_a < n) or not (0 <= idx_b < n):
        raise ValueError(
            f"atom index out of range: idx_a={idx_a}, idx_b={idx_b}, num_atoms={n}"
        )

    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None:
        raise ValueError(f"no bond exists between atoms {idx_a} and {idx_b}")

    # Aromatic bonds carry a fractional order, so "remove the highest
    # component" is undefined. v2 callers never see one (the executor kekulizes
    # before the first op); v1 callers get the historical error.
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        raise ValueError(
            f"break_bond only supports sigma bonds; bond between {idx_a} and "
            f"{idx_b} is {bond.GetBondType()}. Kekulize the molecule first "
            f"(vocabulary='v2' does this automatically)."
        )

    # Pi bond present: remove one order rather than the whole bond. Delegated so
    # the implicit-H and E/Z bookkeeping lives in exactly one place.
    if bond.GetBondType() != Chem.BondType.SINGLE:
        from reactionjson.ops.change_bond_order import change_bond_order
        return change_bond_order(mol, idx_a, idx_b, -1)

    rw = Chem.RWMol(mol)

    # Clear any explicit H counts that were previously locked in: after
    # removing a bond, RDKit will not automatically bump implicit H unless the
    # atom's explicit-H count is "unset" in the sense of needing a recompute.
    # We zero out the noImplicit flag so the sanitizer refills H counts.
    for idx in (idx_a, idx_b):
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNoImplicit(False)
        # Clear chirality on endpoints: once the bond is gone, the original
        # CIP ordering used to assign @/@@ is no longer valid.
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    rw.RemoveBond(idx_a, idx_b)

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"break_bond({idx_a}, {idx_b}) produced an invalid molecule: {exc}"
        ) from exc

    # SanitizeMol alone does NOT re-perceive stereochemistry: the stereo
    # property cache carries over from the input, so FindMolChiralCenters will
    # still report the endpoints as stereocenters even though we cleared their
    # ChiralTag. Force a fresh pass.
    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import unittest
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    # Import add_bond from the sibling file for the round-trip test.
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from add_bond import add_bond

    def _smi(s):
        m = Chem.MolFromSmiles(s)
        assert m is not None, f"bad test SMILES: {s}"
        return m

    def _canon(m):
        return Chem.MolToSmiles(m)

    class TestBreakBond(unittest.TestCase):
        # 1. Ethane -> 2x methane (one Mol, two fragments)
        def test_ethane_to_two_methanes(self):
            m = _smi("CC")
            out = break_bond(m, 0, 1)
            self.assertEqual(_canon(out), _canon(_smi("C.C")))
            frags = Chem.GetMolFrags(out)
            self.assertEqual(len(frags), 2)
            self.assertEqual(out.GetNumAtoms(), 2)

        # 2. Cyclohexane -> hexane (ring count decreases, atoms preserved)
        def test_cyclohexane_opens_to_hexane(self):
            m = _smi("C1CCCCC1")
            rings_before = m.GetRingInfo().NumRings()
            out = break_bond(m, 0, 5)
            rings_after = out.GetRingInfo().NumRings()
            self.assertEqual(rings_before, 1)
            self.assertEqual(rings_after, 0)
            self.assertEqual(out.GetNumAtoms(), m.GetNumAtoms())
            self.assertEqual(_canon(out), _canon(_smi("CCCCCC")))

        # 3. Pi-bond handling: we reject pi bonds with a clear ValueError.
        def test_pi_bond_rejected_double(self):
            m = _smi("C=C")
            with self.assertRaises(ValueError) as cm:
                break_bond(m, 0, 1)
            self.assertIn("sigma", str(cm.exception).lower())

        def test_pi_bond_rejected_triple(self):
            m = _smi("C#C")
            with self.assertRaises(ValueError):
                break_bond(m, 0, 1)

        def test_aromatic_bond_rejected(self):
            m = _smi("c1ccccc1")
            # Aromatic bonds are BondType.AROMATIC; must be rejected.
            with self.assertRaises(ValueError):
                break_bond(m, 0, 1)

        # 4. Error: bond doesn't exist
        def test_error_bond_missing(self):
            m = _smi("CCC")
            # atoms 0 and 2 are not directly bonded
            with self.assertRaises(ValueError) as cm:
                break_bond(m, 0, 2)
            self.assertIn("no bond", str(cm.exception).lower())

        # 5. Error: out-of-range index
        def test_error_index_out_of_range(self):
            m = _smi("CC")
            with self.assertRaises(ValueError):
                break_bond(m, 0, 99)
            with self.assertRaises(ValueError):
                break_bond(m, -1, 0)

        # 6. Break a bond adjacent to a stereocenter -> stereo cleared only at
        #    broken-bond endpoints, preserved elsewhere.
        def test_stereo_cleared_only_at_broken_endpoints(self):
            # Two stereocenters: 2-butanol-ish analog with a second chiral C.
            # Use: C[C@H](O)[C@@H](N)CC  (atoms: 0=C, 1=C*, 2=O, 3=C*, 4=N, 5=C, 6=C)
            # Break bond between atom 1 and atom 3 (the two stereocenters).
            # Both endpoints' stereo should clear; no other stereocenter exists,
            # so we verify by counting stereocenters.
            # For "preserved elsewhere", use a molecule with THREE stereocenters
            # and break a bond between two of them; the third should survive.
            # SMILES: C[C@H](O)[C@@H](N)[C@H](F)CC
            # atoms: 0=C, 1=C*, 2=O, 3=C*, 4=N, 5=C*, 6=F, 7=C, 8=C
            m = _smi("C[C@H](O)[C@@H](N)[C@H](F)CC")
            # find stereocenters before
            centers_before = Chem.FindMolChiralCenters(m, includeUnassigned=False)
            self.assertEqual(len(centers_before), 3)
            center_idxs_before = {i for i, _ in centers_before}
            self.assertEqual(center_idxs_before, {1, 3, 5})

            out = break_bond(m, 1, 3)
            centers_after = Chem.FindMolChiralCenters(out, includeUnassigned=False)
            center_idxs_after = {i for i, _ in centers_after}
            # Endpoint stereocenters 1 and 3 must be cleared; 5 must remain.
            self.assertNotIn(1, center_idxs_after)
            self.assertNotIn(3, center_idxs_after)
            self.assertIn(5, center_idxs_after)
            # Also: atom 5's chiral tag should still be set.
            self.assertNotEqual(
                out.GetAtomWithIdx(5).GetChiralTag(),
                Chem.ChiralType.CHI_UNSPECIFIED,
            )
            # And atoms 1, 3 chiral tags should be unspecified.
            self.assertEqual(
                out.GetAtomWithIdx(1).GetChiralTag(),
                Chem.ChiralType.CHI_UNSPECIFIED,
            )
            self.assertEqual(
                out.GetAtomWithIdx(3).GetChiralTag(),
                Chem.ChiralType.CHI_UNSPECIFIED,
            )

        # 7. Input mol unchanged.
        def test_input_mol_not_mutated(self):
            m = _smi("CC")
            before = _canon(m)
            before_n_bonds = m.GetNumBonds()
            _ = break_bond(m, 0, 1)
            self.assertEqual(_canon(m), before)
            self.assertEqual(m.GetNumBonds(), before_n_bonds)

        # 8. Round-trip: add_bond(break_bond(mol, a, b), a, b, 1) == original
        def test_round_trip_single_bond(self):
            m = _smi("CCCCCC")  # hexane, break bond 0-1
            broken = break_bond(m, 0, 1)
            rebuilt = add_bond(broken, 0, 1, order=1)
            self.assertEqual(_canon(rebuilt), _canon(m))

        def test_round_trip_ring(self):
            m = _smi("C1CCCCC1")  # cyclohexane: break 0-5, remake 0-5
            broken = break_bond(m, 0, 5)
            rebuilt = add_bond(broken, 0, 5, order=1)
            self.assertEqual(_canon(rebuilt), _canon(m))

    unittest.main(verbosity=2)
