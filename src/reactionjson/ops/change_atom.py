"""Atomic op: change_atom.

Transmutes the element of a single atom in a molecule (e.g. C -> N), letting
RDKit's sanitizer recompute implicit hydrogens for the new element's default
valence. Returns a new sanitized Mol; the input is not mutated.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


def change_atom(mol: Chem.Mol, idx: int, symbol: str) -> Chem.Mol:
    """Change the element of atom idx to `symbol` (e.g. 'C' -> 'N').

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - idx is out of range
        - symbol is not a known element
        - the resulting molecule fails sanitization (valence violation,
          aromaticity broken, etc.)
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")

    # RDKit raises RuntimeError (not returns 0) on unknown symbols.
    pt = Chem.GetPeriodicTable()
    try:
        atomic_num = pt.GetAtomicNumber(symbol)
    except RuntimeError as exc:
        raise ValueError(f"unknown element symbol: {symbol!r}") from exc
    if atomic_num <= 0:
        raise ValueError(f"unknown element symbol: {symbol!r}")

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)

    atom.SetAtomicNum(atomic_num)
    # Implicit-H auto-adjust is unreliable across transmutations. Reset the
    # H-related flags so the sanitizer recomputes H count from the new
    # element's default valence.
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"change_atom({idx}, {symbol!r}) produced an invalid molecule: {exc}"
        ) from exc

    # SanitizeMol does not re-perceive stereochemistry; transmutation can
    # create or destroy stereocenters, so force a fresh pass.
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

    class TestChangeAtom(unittest.TestCase):
        # 1. Methane C -> N (CH4 -> NH3 via implicit-H recompute)
        def test_methane_to_ammonia(self):
            m = _smi("C")
            out = change_atom(m, 0, "N")
            self.assertEqual(_canon(out), _canon(_smi("N")))
            self.assertEqual(out.GetAtomWithIdx(0).GetSymbol(), "N")
            # NH3: 3 implicit Hs
            self.assertEqual(out.GetAtomWithIdx(0).GetTotalNumHs(), 3)

        # 2. C -> N on a reasonable case (methanol's methyl C -> N -> methylamine-oxide-ish;
        #    more cleanly: ethanol's CH3 (atom 0) -> N gives aminomethanol H2N-CH2-OH...
        #    use ethane CC: atom 0 C -> N yields CN (methylamine).
        def test_ethane_c_to_n_gives_methylamine(self):
            m = _smi("CC")
            out = change_atom(m, 0, "N")
            self.assertEqual(_canon(out), _canon(_smi("CN")))

        # 3. Non-aromatic N -> C in an amine: methylamine CN -> ethane CC.
        def test_amine_n_to_c(self):
            m = _smi("CN")
            out = change_atom(m, 1, "C")
            self.assertEqual(_canon(out), _canon(_smi("CC")))

        # 4. Error: idx out of range
        def test_error_idx_out_of_range(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                change_atom(m, 99, "N")
            self.assertIn("out of range", str(cm.exception))
            with self.assertRaises(ValueError):
                change_atom(m, -1, "N")

        # 5. Error: unknown symbol
        def test_error_unknown_symbol(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                change_atom(m, 0, "Xx")
            self.assertIn("unknown", str(cm.exception).lower())

        # 6. Error: valence violation. Neopentane CC(C)(C)C has a quaternary C
        #    at atom 1 with 4 bonds to heavy atoms. Changing it to O (valence 2)
        #    would require removing 2 bonds -- sanitize should reject.
        def test_error_valence_violation(self):
            m = _smi("CC(C)(C)C")
            # atom 1 is the quaternary C (4 bonds to heavy atoms)
            self.assertEqual(m.GetAtomWithIdx(1).GetDegree(), 4)
            with self.assertRaises(ValueError) as cm:
                change_atom(m, 1, "O")
            self.assertIn("invalid", str(cm.exception).lower())

        # 7. Error: aromaticity break. Benzene c1ccccc1; C -> He fails
        #    sanitization because He has no accepted valence. (RDKit accepts
        #    C->Si as silabenzene; He/Ne/Na clearly cannot sit in the ring.)
        def test_error_aromaticity_break(self):
            m = _smi("c1ccccc1")
            with self.assertRaises(ValueError) as cm:
                change_atom(m, 0, "He")
            self.assertIn("invalid", str(cm.exception).lower())

        # 8. Input mol unchanged
        def test_input_mol_not_mutated(self):
            m = _smi("CC")
            before = _canon(m)
            before_sym_0 = m.GetAtomWithIdx(0).GetSymbol()
            _ = change_atom(m, 0, "N")
            self.assertEqual(_canon(m), before)
            self.assertEqual(m.GetAtomWithIdx(0).GetSymbol(), before_sym_0)

        # 9. Preserves stereochemistry on other atoms. Use a molecule with a
        #    stereocenter that is NOT the transmuted atom, and change a distant
        #    atom. Start: C[C@H](O)CC -- atom 1 is the stereocenter, atom 3/4
        #    are the ethyl tail. Change atom 4 (terminal C) to N. Stereo on
        #    atom 1 must survive.
        def test_preserves_distant_stereo(self):
            m = _smi("C[C@H](O)CC")
            centers_before = Chem.FindMolChiralCenters(m, includeUnassigned=False)
            self.assertEqual(len(centers_before), 1)
            self.assertEqual(centers_before[0][0], 1)
            tag_before = m.GetAtomWithIdx(1).GetChiralTag()

            out = change_atom(m, 4, "N")
            self.assertEqual(out.GetAtomWithIdx(4).GetSymbol(), "N")
            # stereocenter at atom 1 should still be there and assigned.
            centers_after = Chem.FindMolChiralCenters(out, includeUnassigned=False)
            center_idxs_after = {i for i, _ in centers_after}
            self.assertIn(1, center_idxs_after)
            self.assertEqual(out.GetAtomWithIdx(1).GetChiralTag(), tag_before)

        # 10. SMILES reflects the change: CCC (propane) atom 1 (middle C) -> N
        #     gives CNC (dimethylamine).
        def test_smiles_reflects_change(self):
            m = _smi("CCC")
            out = change_atom(m, 1, "N")
            self.assertEqual(_canon(out), _canon(_smi("CNC")))
            self.assertEqual(out.GetAtomWithIdx(1).GetSymbol(), "N")

    unittest.main(verbosity=2)
