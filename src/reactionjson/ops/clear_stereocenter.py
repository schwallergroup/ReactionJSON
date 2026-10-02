"""Atomic op: clear_stereocenter.

Clears the chiral tag at a single atom (sets it to CHI_UNSPECIFIED), returning
a new sanitized Mol. Idempotent: calling on an already-unspecified atom is a
no-op. Stereochemistry elsewhere in the molecule is preserved.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


def clear_stereocenter(mol: Chem.Mol, idx: int) -> Chem.Mol:
    """Clear stereochemistry at atom idx (set to CHI_UNSPECIFIED). Idempotent.

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError only if idx is out of range.
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")

    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"clear_stereocenter({idx}) produced an invalid molecule: {exc}"
        ) from exc

    # SanitizeMol does NOT re-perceive stereochemistry. Without this pass the
    # cached CIP codes/stereo flags survive and the @/@@ marker stays in the
    # canonical SMILES even though the ChiralTag is unspecified.
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

    class TestClearStereocenter(unittest.TestCase):
        # 1. (R)-alanine: clear the central stereocenter -> no @ in SMILES.
        def test_alanine_stereocenter_cleared(self):
            # C[C@@H](N)C(=O)O: atom 0=C (CH3), 1=C* (chiral), 2=N, 3=C, ...
            m = _smi("C[C@@H](N)C(=O)O")
            self.assertIn(m.GetAtomWithIdx(1).GetChiralTag(),
                          (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                           Chem.ChiralType.CHI_TETRAHEDRAL_CCW))
            out = clear_stereocenter(m, 1)
            smi = _canon(out)
            self.assertNotIn("@", smi)
            self.assertEqual(smi, _canon(_smi("CC(N)C(=O)O")))
            self.assertEqual(out.GetAtomWithIdx(1).GetChiralTag(),
                             Chem.ChiralType.CHI_UNSPECIFIED)

        # 2. Idempotent: clearing an already-unspecified atom is a no-op, no error.
        def test_idempotent_on_unspecified_atom(self):
            m = _smi("C[C@@H](N)C(=O)O")
            # atom 0 is the CH3 carbon, already unspecified.
            self.assertEqual(m.GetAtomWithIdx(0).GetChiralTag(),
                             Chem.ChiralType.CHI_UNSPECIFIED)
            out = clear_stereocenter(m, 0)
            # Mol must remain valid; the real stereocenter at atom 1 should
            # still be present (we only cleared atom 0's tag, which was none).
            self.assertEqual(_canon(out), _canon(m))
            self.assertEqual(out.GetAtomWithIdx(0).GetChiralTag(),
                             Chem.ChiralType.CHI_UNSPECIFIED)

        # 2b. Idempotent: clearing an already-cleared stereocenter twice.
        def test_idempotent_double_clear(self):
            m = _smi("C[C@@H](N)C(=O)O")
            once = clear_stereocenter(m, 1)
            twice = clear_stereocenter(once, 1)
            self.assertEqual(_canon(once), _canon(twice))

        # 3. Error: idx out of range.
        def test_error_index_out_of_range(self):
            m = _smi("C[C@@H](N)C(=O)O")
            with self.assertRaises(ValueError):
                clear_stereocenter(m, 99)
            with self.assertRaises(ValueError):
                clear_stereocenter(m, -1)

        # 4. Preserves other stereocenters.
        def test_preserves_other_stereocenters(self):
            # Two stereocenters: 2,3-dihydroxybutane-ish -- use a clear double
            # stereocenter SMILES. C[C@H](O)[C@@H](O)C
            # atoms: 0=C, 1=C*, 2=O, 3=C*, 4=O, 5=C
            m = _smi("C[C@H](O)[C@@H](O)C")
            centers_before = Chem.FindMolChiralCenters(m, includeUnassigned=False)
            idxs_before = {i for i, _ in centers_before}
            self.assertEqual(idxs_before, {1, 3})

            out = clear_stereocenter(m, 1)

            # Output SMILES must still contain an @ marker (the surviving one).
            self.assertIn("@", _canon(out))

            centers_after = Chem.FindMolChiralCenters(out, includeUnassigned=False)
            idxs_after = {i for i, _ in centers_after}
            self.assertNotIn(1, idxs_after)
            self.assertIn(3, idxs_after)

        # 5. Input mol unchanged.
        def test_input_mol_not_mutated(self):
            m = _smi("C[C@@H](N)C(=O)O")
            before_smi = _canon(m)
            before_tag = m.GetAtomWithIdx(1).GetChiralTag()
            _ = clear_stereocenter(m, 1)
            self.assertEqual(_canon(m), before_smi)
            self.assertEqual(m.GetAtomWithIdx(1).GetChiralTag(), before_tag)

        # 6. FindMolChiralCenters no longer lists the cleared atom.
        def test_find_chiral_centers_drops_cleared_idx(self):
            m = _smi("C[C@@H](N)C(=O)O")
            centers_before = Chem.FindMolChiralCenters(m, includeUnassigned=False)
            self.assertIn(1, {i for i, _ in centers_before})
            out = clear_stereocenter(m, 1)
            centers_after = Chem.FindMolChiralCenters(out, includeUnassigned=False)
            self.assertNotIn(1, {i for i, _ in centers_after})

    unittest.main(verbosity=2)
