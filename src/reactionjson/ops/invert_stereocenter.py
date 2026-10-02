"""Atomic op: invert_stereocenter.

Flips the chiral tag (CW <-> CCW) at a tetrahedral stereocenter, returning a
new sanitized Mol with re-perceived stereochemistry. Only atoms that already
carry a tetrahedral chiral tag are accepted; unspecified or non-tetrahedral
chirality is out of scope for this op.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


_FLIP = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
}


def invert_stereocenter(mol: Chem.Mol, idx: int) -> Chem.Mol:
    """Invert the chiral tag at atom idx (CW <-> CCW).

    Returns a new sanitized Mol (input is not mutated) with stereo re-perceived
    so that downstream CIP labels (R/S via FindMolChiralCenters) reflect the
    inversion.

    Raises ValueError if:
        - idx is out of range
        - the atom has CHI_UNSPECIFIED (not a stereocenter)
        - the atom has CHI_OTHER or any non-tetrahedral chirality
        - the resulting molecule fails sanitization
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")

    tag = mol.GetAtomWithIdx(idx).GetChiralTag()
    if tag == Chem.ChiralType.CHI_UNSPECIFIED:
        raise ValueError(
            f"atom {idx} has CHI_UNSPECIFIED: not a stereocenter. Set a chiral "
            f"tag first or use a different op."
        )
    if tag not in _FLIP:
        raise ValueError(
            f"atom {idx} has non-tetrahedral chirality {tag!r}; out of scope "
            f"for invert_stereocenter."
        )

    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(idx).SetChiralTag(_FLIP[tag])

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"invert_stereocenter({idx}) produced an invalid molecule: {exc}"
        ) from exc

    # The CW/CCW tag is relative to the neighbor order in the current SMILES
    # traversal; after flipping, round-trip through canonical SMILES so the
    # tag is re-normalized against the canonical atom ordering.
    canon = Chem.MolToSmiles(new_mol)
    new_mol = Chem.MolFromSmiles(canon)
    if new_mol is None:
        raise ValueError(
            f"invert_stereocenter({idx}): canonical round-trip failed for {canon!r}"
        )

    # SanitizeMol does NOT re-perceive stereochemistry; force a fresh pass so
    # that CIP labels (R/S) on FindMolChiralCenters reflect the inversion.
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

    def _center_idx_for_atom_map(m, target_canonical_atom):
        """Find the atom index in m whose (symbol, degree) matches target."""
        for a in m.GetAtoms():
            if a.GetSymbol() == target_canonical_atom[0] and a.GetDegree() == target_canonical_atom[1]:
                if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                    return a.GetIdx()
        return None

    class TestInvertStereocenter(unittest.TestCase):
        # 1. (R)-alanine -> (S)-alanine via CIP label.
        def test_R_alanine_inverts_to_S_alanine(self):
            m = _smi("C[C@@H](N)C(=O)O")  # (S)-alanine in RDKit conv? verify below
            centers_before = dict(
                Chem.FindMolChiralCenters(m, includeUnassigned=False)
            )
            # Central C is atom 1.
            self.assertIn(1, centers_before)
            label_before = centers_before[1]
            self.assertIn(label_before, ("R", "S"))

            out = invert_stereocenter(m, 1)
            # The inverted mol was re-canonicalized; locate the stereocenter
            # again by CIP scan.
            centers_after = dict(
                Chem.FindMolChiralCenters(out, includeUnassigned=False)
            )
            self.assertEqual(len(centers_after), 1)
            label_after = list(centers_after.values())[0]
            self.assertIn(label_after, ("R", "S"))
            self.assertNotEqual(label_before, label_after)

        # 2. Round-trip: invert twice -> original canonical SMILES.
        def test_double_invert_round_trip(self):
            m = _smi("C[C@@H](N)C(=O)O")
            once = invert_stereocenter(m, 1)
            # Re-locate stereocenter index in the canonical `once` mol.
            centers = Chem.FindMolChiralCenters(once, includeUnassigned=False)
            self.assertEqual(len(centers), 1)
            center_idx = centers[0][0]
            twice = invert_stereocenter(once, center_idx)
            self.assertEqual(_canon(twice), _canon(m))

        # 3. Error: idx out of range.
        def test_error_idx_out_of_range(self):
            m = _smi("C[C@@H](N)C(=O)O")
            with self.assertRaises(ValueError):
                invert_stereocenter(m, 99)
            with self.assertRaises(ValueError):
                invert_stereocenter(m, -1)

        # 4. Error: atom is not a stereocenter (CHI_UNSPECIFIED) — e.g. the
        #    methyl carbon (atom 0) of alanine.
        def test_error_unspecified_stereocenter(self):
            m = _smi("C[C@@H](N)C(=O)O")
            self.assertEqual(
                m.GetAtomWithIdx(0).GetChiralTag(),
                Chem.ChiralType.CHI_UNSPECIFIED,
            )
            with self.assertRaises(ValueError) as cm:
                invert_stereocenter(m, 0)
            self.assertIn("unspecified", str(cm.exception).lower())

        # 5. Preserves other stereocenters. Use L-threonine-ish:
        #    C[C@H](O)[C@@H](N)C(=O)O -- two stereocenters at atoms 1 and 3.
        def test_preserves_other_stereocenters(self):
            m = _smi("C[C@H](O)[C@@H](N)C(=O)O")
            centers_before = dict(
                Chem.FindMolChiralCenters(m, includeUnassigned=False)
            )
            self.assertEqual(len(centers_before), 2)
            self.assertIn(1, centers_before)
            self.assertIn(3, centers_before)
            label_1_before = centers_before[1]
            label_3_before = centers_before[3]

            out = invert_stereocenter(m, 1)
            # Re-canonicalization may renumber atoms. Identify the two centers
            # in `out` by matching neighbor composition.
            def _neighbors_multiset(mol, idx):
                atom = mol.GetAtomWithIdx(idx)
                return tuple(sorted(n.GetSymbol() for n in atom.GetNeighbors()))

            # In the input, atom 1's neighbors are {C, C, O} and atom 3's are
            # {C, C, N}. These are distinguishable.
            orig_1_sig = _neighbors_multiset(m, 1)
            orig_3_sig = _neighbors_multiset(m, 3)
            self.assertNotEqual(orig_1_sig, orig_3_sig)

            centers_after = dict(
                Chem.FindMolChiralCenters(out, includeUnassigned=False)
            )
            self.assertEqual(len(centers_after), 2)

            inverted_label = None
            untouched_label = None
            for idx_after, lab in centers_after.items():
                sig = _neighbors_multiset(out, idx_after)
                if sig == orig_1_sig:
                    inverted_label = lab
                elif sig == orig_3_sig:
                    untouched_label = lab

            self.assertIsNotNone(inverted_label)
            self.assertIsNotNone(untouched_label)
            # The inverted center's CIP label must flip.
            self.assertNotEqual(inverted_label, label_1_before)
            # The untouched center's CIP label must be identical.
            self.assertEqual(untouched_label, label_3_before)

        # 6. Input mol unchanged.
        def test_input_mol_not_mutated(self):
            m = _smi("C[C@@H](N)C(=O)O")
            before = _canon(m)
            before_tag = m.GetAtomWithIdx(1).GetChiralTag()
            _ = invert_stereocenter(m, 1)
            self.assertEqual(_canon(m), before)
            self.assertEqual(m.GetAtomWithIdx(1).GetChiralTag(), before_tag)

        # 7. Non-chiral atoms elsewhere in the mol are not affected. Inverting
        #    one stereocenter should leave every CHI_UNSPECIFIED atom still
        #    CHI_UNSPECIFIED (no spurious chirality introduced by the op).
        def test_non_chiral_atoms_unaffected(self):
            m = _smi("C[C@H](O)[C@@H](N)C(=O)O")
            unspec_before = {
                a.GetIdx()
                for a in m.GetAtoms()
                if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
            }

            out = invert_stereocenter(m, 1)

            # Count unspecified atoms in `out`; the count should be preserved
            # (chirality count is invariant under inversion of one tag).
            unspec_after = {
                a.GetIdx()
                for a in out.GetAtoms()
                if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
            }
            self.assertEqual(len(unspec_before), len(unspec_after))
            # Stereocenter count also preserved.
            centers_before = Chem.FindMolChiralCenters(m, includeUnassigned=False)
            centers_after = Chem.FindMolChiralCenters(out, includeUnassigned=False)
            self.assertEqual(len(centers_before), len(centers_after))

        # Extra: CHI_OTHER is rejected. We can't easily construct CHI_OTHER
        # from SMILES, so we synthesize by setting the tag directly on a mol.
        def test_error_chi_other_rejected(self):
            m = _smi("C[C@@H](N)C(=O)O")
            rw = Chem.RWMol(m)
            rw.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_OTHER)
            m2 = rw.GetMol()
            _sanitize(m2)
            with self.assertRaises(ValueError) as cm:
                invert_stereocenter(m2, 1)
            self.assertIn("non-tetrahedral", str(cm.exception).lower())

    unittest.main(verbosity=2)
