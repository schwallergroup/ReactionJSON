"""Atomic op: set_bond_stereo.

Sets E/Z stereochemistry on a double bond between two atoms. Requires a
reference neighbor on each end (auto-selected if not provided). Returns a new
sanitized Mol with the stereo assignment encoded so that canonical SMILES
output contains the `/` and `\\` bond-direction markers.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


_STEREO_MAP = {
    "E": Chem.BondStereo.STEREOE,
    "Z": Chem.BondStereo.STEREOZ,
}


def set_bond_stereo(
    mol: Chem.Mol,
    idx_a: int,
    idx_b: int,
    stereo: str,
    ref_a: int | None = None,
    ref_b: int | None = None,
) -> Chem.Mol:
    """Set E/Z stereo on a double bond between idx_a and idx_b.

    stereo: 'E' or 'Z' (case-insensitive).
    ref_a: neighbor of idx_a (other than idx_b) used as the reference for
        stereo. If None, the first non-idx_b neighbor of idx_a is used.
    ref_b: same for idx_b.

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - either index is out of range
        - no bond exists between idx_a and idx_b
        - the bond is not DOUBLE
        - stereo is not in {'E', 'e', 'Z', 'z'}
        - a ref index is supplied but is not a neighbor of its endpoint
        - no valid reference neighbor exists on an end (terminal =CH2)
    """
    if not isinstance(stereo, str):
        raise ValueError(f"stereo must be 'E' or 'Z'; got {stereo!r}")
    stereo_norm = stereo.upper()
    if stereo_norm not in _STEREO_MAP:
        raise ValueError(f"stereo must be 'E' or 'Z'; got {stereo!r}")

    n = mol.GetNumAtoms()
    if not (0 <= idx_a < n) or not (0 <= idx_b < n):
        raise ValueError(
            f"atom index out of range: idx_a={idx_a}, idx_b={idx_b}, num_atoms={n}"
        )

    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None:
        raise ValueError(f"no bond exists between atoms {idx_a} and {idx_b}")
    if bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError(
            f"set_bond_stereo requires a DOUBLE bond; bond between {idx_a} and "
            f"{idx_b} is {bond.GetBondType()}"
        )

    def _pick_ref(endpoint: int, partner: int, supplied: int | None) -> int:
        atom = mol.GetAtomWithIdx(endpoint)
        neighbor_idxs = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != partner]
        if supplied is not None:
            if supplied not in neighbor_idxs:
                raise ValueError(
                    f"ref atom {supplied} is not a neighbor of atom {endpoint} "
                    f"(neighbors excluding partner {partner}: {neighbor_idxs})"
                )
            return supplied
        if not neighbor_idxs:
            raise ValueError(
                f"atom {endpoint} has no neighbor other than {partner}; cannot "
                f"define E/Z stereo on a terminal double bond"
            )
        return neighbor_idxs[0]

    chosen_ref_a = _pick_ref(idx_a, idx_b, ref_a)
    chosen_ref_b = _pick_ref(idx_b, idx_a, ref_b)

    rw = Chem.RWMol(mol)
    new_bond = rw.GetBondBetweenAtoms(idx_a, idx_b)
    # Stereo atoms are given in the bond's begin/end order, which need not be
    # (idx_a, idx_b). E/Z is symmetric in the two ends, so swapping is safe.
    if new_bond.GetBeginAtomIdx() != idx_a:
        chosen_ref_a, chosen_ref_b = chosen_ref_b, chosen_ref_a
    # SetStereoAtoms must precede SetStereo; STEREOE/STEREOZ are defined
    # *relative to* these two reference atoms.
    new_bond.SetStereoAtoms(chosen_ref_a, chosen_ref_b)
    new_bond.SetStereo(_STEREO_MAP[stereo_norm])

    new_mol = rw.GetMol()
    try:
        # cleanIt=False is critical: AssignStereochemistry(cleanIt=True,
        # force=True) would recompute bond stereo from atomic coordinates or
        # from `/`/`\` bond dirs (neither present here) and wipe the STEREOE/
        # STEREOZ we just set. Plain SanitizeMol preserves it.
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"set_bond_stereo({idx_a}, {idx_b}, {stereo!r}) produced an "
            f"invalid molecule: {exc}"
        ) from exc

    # Canonical SMILES emits `/` and `\` from bond-direction flags on the
    # single bonds adjacent to the double bond. SetDoubleBondNeighborDirections
    # writes those flags from the STEREOE/STEREOZ we just set on the double
    # bond, without clobbering it. (SetBondStereoFromDirections goes the wrong
    # way — it reads directions and writes stereo, which would reset us.)
    Chem.SetDoubleBondNeighborDirections(new_mol)

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

    class TestSetBondStereo(unittest.TestCase):
        # 1. 2-butene: set E -> canonical SMILES C/C=C/C
        def test_2butene_set_E(self):
            m = _smi("CC=CC")  # atoms: 0=C, 1=C, 2=C, 3=C; double bond 1-2
            out = set_bond_stereo(m, 1, 2, "E")
            smi = _canon(out)
            self.assertIn("/", smi)
            # E-2-butene canonicalizes to "C/C=C/C"
            self.assertEqual(smi, _canon(_smi("C/C=C/C")))

        # 2. 2-butene: set Z -> canonical SMILES C/C=C\C
        def test_2butene_set_Z(self):
            m = _smi("CC=CC")
            out = set_bond_stereo(m, 1, 2, "Z")
            smi = _canon(out)
            self.assertTrue("/" in smi and "\\" in smi)
            self.assertEqual(smi, _canon(_smi("C/C=C\\C")))

        # 3. Explicit refs honored
        def test_explicit_refs(self):
            m = _smi("CC=CC")
            out = set_bond_stereo(m, 1, 2, "E", ref_a=0, ref_b=3)
            self.assertEqual(_canon(out), _canon(_smi("C/C=C/C")))

        # 4. Lowercase 'e' / 'z' accepted
        def test_lowercase_stereo(self):
            m = _smi("CC=CC")
            out_e = set_bond_stereo(m, 1, 2, "e")
            out_z = set_bond_stereo(m, 1, 2, "z")
            self.assertEqual(_canon(out_e), _canon(_smi("C/C=C/C")))
            self.assertEqual(_canon(out_z), _canon(_smi("C/C=C\\C")))

        # 5. Error: bond is single
        def test_error_single_bond(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                set_bond_stereo(m, 0, 1, "E")
            self.assertIn("DOUBLE", str(cm.exception))

        # 6. Error: bond doesn't exist
        def test_error_bond_missing(self):
            m = _smi("CCC")
            with self.assertRaises(ValueError) as cm:
                set_bond_stereo(m, 0, 2, "E")
            self.assertIn("no bond", str(cm.exception).lower())

        # 7. Error: stereo is 'X'
        def test_error_bad_stereo(self):
            m = _smi("CC=CC")
            with self.assertRaises(ValueError) as cm:
                set_bond_stereo(m, 1, 2, "X")
            self.assertIn("E", str(cm.exception))

        # 8. Error: ref_a not a neighbor of idx_a
        def test_error_ref_not_neighbor(self):
            m = _smi("CC=CC")
            # atom 3 is not a neighbor of atom 1
            with self.assertRaises(ValueError) as cm:
                set_bond_stereo(m, 1, 2, "E", ref_a=3)
            self.assertIn("neighbor", str(cm.exception).lower())

        # 9. Error: terminal alkene C=C has no ref on either end
        def test_error_terminal_alkene(self):
            m = _smi("C=C")
            with self.assertRaises(ValueError) as cm:
                set_bond_stereo(m, 0, 1, "E")
            self.assertIn("terminal", str(cm.exception).lower())

        # 10. Input mol not mutated
        def test_input_mol_not_mutated(self):
            m = _smi("CC=CC")
            before = _canon(m)
            before_bond_stereo = m.GetBondBetweenAtoms(1, 2).GetStereo()
            _ = set_bond_stereo(m, 1, 2, "E")
            self.assertEqual(_canon(m), before)
            self.assertEqual(
                m.GetBondBetweenAtoms(1, 2).GetStereo(), before_bond_stereo
            )

        # 11. Round-trip: set E then set Z changes the stereo
        def test_round_trip_E_then_Z(self):
            m = _smi("CC=CC")
            e_mol = set_bond_stereo(m, 1, 2, "E")
            z_mol = set_bond_stereo(e_mol, 1, 2, "Z")
            self.assertNotEqual(_canon(e_mol), _canon(z_mol))
            self.assertEqual(_canon(e_mol), _canon(_smi("C/C=C/C")))
            self.assertEqual(_canon(z_mol), _canon(_smi("C/C=C\\C")))

    unittest.main(verbosity=2)
