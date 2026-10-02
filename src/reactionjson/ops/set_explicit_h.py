"""Atomic op: set_explicit_h.

Sets the number of explicit hydrogens on a single atom to exactly ``n`` and
locks the H count (``noImplicit=True``) so later sanitization will not silently
re-add or remove Hs. Returns a new sanitized Mol. Raises ValueError on any
out-of-range index, negative n, or valence violation.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


def set_explicit_h(mol: Chem.Mol, idx: int, n: int) -> Chem.Mol:
    """Set the number of explicit hydrogens on atom idx to exactly n.

    Returns a new sanitized Mol (input is not mutated). The target atom has
    ``NoImplicit=True`` set after this op, so subsequent edits will not silently
    adjust its H count.

    Raises ValueError if:
        - idx is out of range
        - n < 0
        - the resulting molecule fails sanitization (e.g. valence violation)
    """
    num_atoms = mol.GetNumAtoms()
    if not (0 <= idx < num_atoms):
        raise ValueError(
            f"atom index out of range: idx={idx}, num_atoms={num_atoms}"
        )
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")

    rw = Chem.RWMol(mol)

    # Kekulize before touching an aromatic atom: changing the H count on an
    # aromatic ring atom can invalidate the aromatic perception, which then
    # fails kekulization on the way back out. Kekulizing up-front makes the
    # bond orders explicit so sanitization can re-perceive aromaticity cleanly.
    atom = rw.GetAtomWithIdx(idx)
    if atom.GetIsAromatic():
        Chem.Kekulize(rw, clearAromaticFlags=True)
        atom = rw.GetAtomWithIdx(idx)

    atom.SetNumExplicitHs(n)
    atom.SetNoImplicit(True)

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"set_explicit_h(idx={idx}, n={n}) produced an invalid molecule: {exc}"
        ) from exc

    # SanitizeMol does NOT re-perceive stereochemistry; changing H count can
    # promote/demote a stereocenter (e.g. CHRR' vs CH2R), so force a fresh pass.
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

    class TestSetExplicitH(unittest.TestCase):
        # 1. Increase H count on a carbon: benzene aromatic C -> explicit H=1.
        #    Every aromatic C in benzene already has 1 implicit H; the op locks
        #    it in as explicit. The ring is still aromatic benzene; only atom 0
        #    has its H count locked. (Canonical SMILES may now render atom 0 as
        #    [cH] because noImplicit is a non-default flag, so we check the
        #    structure directly rather than comparing SMILES strings.)
        def test_increase_h_on_aromatic_carbon(self):
            m = _smi("c1ccccc1")
            out = set_explicit_h(m, 0, 1)
            a = out.GetAtomWithIdx(0)
            self.assertEqual(a.GetNumExplicitHs(), 1)
            self.assertTrue(a.GetNoImplicit())
            self.assertEqual(a.GetTotalNumHs(), 1)
            # Ring structure preserved: 6 atoms, all aromatic carbons, 1 ring.
            self.assertEqual(out.GetNumAtoms(), 6)
            self.assertEqual(out.GetRingInfo().NumRings(), 1)
            for atom in out.GetAtoms():
                self.assertEqual(atom.GetSymbol(), "C")
                self.assertTrue(atom.GetIsAromatic())

        # 2. Decrease H count: methane atom 0, set n=3 (free one valence).
        #    The resulting atom is a CH3 "radical-like" with an open valence;
        #    rdkit represents this as a carbon with 3 Hs and one unsatisfied
        #    valence, visible as a radical in the SMILES.
        def test_decrease_h_on_methane(self):
            m = _smi("C")
            out = set_explicit_h(m, 0, 3)
            a = out.GetAtomWithIdx(0)
            self.assertEqual(a.GetNumExplicitHs(), 3)
            self.assertTrue(a.GetNoImplicit())
            self.assertEqual(a.GetTotalNumHs(), 3)
            # SMILES should show 3 Hs on the carbon ([CH3])
            smi = _canon(out)
            self.assertIn("[CH3]", smi)

        # 3. Error: idx out of range
        def test_error_idx_out_of_range(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                set_explicit_h(m, 99, 1)
            self.assertIn("out of range", str(cm.exception))
            with self.assertRaises(ValueError):
                set_explicit_h(m, -1, 1)

        # 4. Error: n < 0
        def test_error_negative_n(self):
            m = _smi("C")
            with self.assertRaises(ValueError) as cm:
                set_explicit_h(m, 0, -1)
            self.assertIn(">= 0", str(cm.exception))

        # 5. Error: valence violation. Neopentane's central C has 4 heavy
        #    bonds already; setting 5 Hs on it pushes total valence to 9.
        def test_error_valence_violation(self):
            m = _smi("CC(C)(C)C")
            # atom 1 is the quaternary central C (4 heavy neighbors)
            self.assertEqual(m.GetAtomWithIdx(1).GetDegree(), 4)
            with self.assertRaises(ValueError) as cm:
                set_explicit_h(m, 1, 5)
            self.assertIn("invalid", str(cm.exception).lower())

        # 6. Input mol unchanged (canonical SMILES preserved, atom props too)
        def test_input_mol_not_mutated(self):
            m = _smi("C")
            before_smi = _canon(m)
            before_noimpl = m.GetAtomWithIdx(0).GetNoImplicit()
            before_nexpl = m.GetAtomWithIdx(0).GetNumExplicitHs()
            _ = set_explicit_h(m, 0, 3)
            self.assertEqual(_canon(m), before_smi)
            self.assertEqual(m.GetAtomWithIdx(0).GetNoImplicit(), before_noimpl)
            self.assertEqual(m.GetAtomWithIdx(0).GetNumExplicitHs(), before_nexpl)

        # 7. Output reflects the H change. Setting n=2 on ethane atom 0 should
        #    leave the carbon with an open valence, visible as [CH2] in SMILES.
        #    (Note: when the locked H count matches the default valence, RDKit
        #    strips brackets from canonical SMILES, so we only assert the visible
        #    bracket case and verify flags/counts on the atom directly.)
        def test_output_smiles_reflects_h_change(self):
            m = _smi("CC")
            self.assertNotIn("[CH2]", _canon(m))
            out = set_explicit_h(m, 0, 2)
            self.assertIn("[CH2]", _canon(out))
            a = out.GetAtomWithIdx(0)
            self.assertEqual(a.GetNumExplicitHs(), 2)
            self.assertTrue(a.GetNoImplicit())
            self.assertEqual(a.GetTotalNumHs(), 2)

    unittest.main(verbosity=2)
