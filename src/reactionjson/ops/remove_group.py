"""Atomic op: remove_group.

Removes a pendant group from a molecule. The bond between the anchor atom
`idx` (which stays) and `root_idx` (the first atom of the group to remove) is
broken, and every atom in the fragment rooted at `root_idx` is deleted. The
anchor's freed valence is refilled with implicit Hs via sanitize.

Only true pendant groups are supported: if the (idx, root_idx) bond is in a
ring, breaking it would not disconnect a fragment, and the op rejects it.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


_ANCHOR_TAG = 999


def remove_group(mol: Chem.Mol, idx: int, root_idx: int) -> Chem.Mol:
    """Remove the pendant group rooted at root_idx, anchored at idx.

    Returns a new sanitized Mol (input not mutated). The anchor `idx` stays;
    every atom in the connected component containing `root_idx` (after breaking
    the idx-root_idx bond) is deleted.

    Raises ValueError if:
        - either index is out of range
        - idx == root_idx
        - no bond exists between idx and root_idx
        - the bond between them is in a ring (not a true pendant)
        - the resulting molecule fails sanitization
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n) or not (0 <= root_idx < n):
        raise ValueError(
            f"atom index out of range: idx={idx}, root_idx={root_idx}, num_atoms={n}"
        )
    if idx == root_idx:
        raise ValueError(f"idx and root_idx must be distinct; got {idx}")

    bond = mol.GetBondBetweenAtoms(idx, root_idx)
    if bond is None:
        raise ValueError(f"no bond exists between atoms {idx} and {root_idx}")

    rw = Chem.RWMol(mol)
    # Ring perception must be current for IsInRing to be reliable on an RWMol.
    Chem.GetSSSR(rw)
    rw_bond = rw.GetBondBetweenAtoms(idx, root_idx)
    if rw_bond.IsInRing():
        raise ValueError(
            f"not a pendant group - bond ({idx}, {root_idx}) is in a ring"
        )

    bond_order = rw_bond.GetBondTypeAsDouble()

    # Persistent anchor: atom indices will shift after RemoveAtom, so tag the
    # anchor with a sentinel atom-map number we can scan for afterward.
    anchor_atom = rw.GetAtomWithIdx(idx)
    prev_map = anchor_atom.GetAtomMapNum()
    anchor_atom.SetAtomMapNum(_ANCHOR_TAG)

    # Break the bond first, then discover which fragment holds root_idx.
    rw.RemoveBond(idx, root_idx)
    frags = Chem.GetMolFrags(rw, asMols=False, sanitizeFrags=False)
    group_frag = None
    for frag in frags:
        if root_idx in frag:
            group_frag = frag
            break
    if group_frag is None:
        raise ValueError(
            f"internal: root_idx={root_idx} not found in any fragment"
        )

    # Remove in descending index order to avoid reindex invalidation.
    for aidx in sorted(group_frag, reverse=True):
        rw.RemoveAtom(aidx)

    # Locate the anchor by sentinel tag, restore its prior map number.
    new_anchor_idx = None
    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() == _ANCHOR_TAG:
            new_anchor_idx = atom.GetIdx()
            atom.SetAtomMapNum(prev_map)
            break
    if new_anchor_idx is None:
        raise ValueError("internal: anchor atom lost during removal")

    anchor = rw.GetAtomWithIdx(new_anchor_idx)
    # The removed neighbor was part of the CIP ordering; clear stereo.
    anchor.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    # Valence refill: if the anchor carries explicit Hs (locked), bump it by
    # the removed bond's order to compensate; otherwise let sanitize recompute.
    if anchor.GetNumExplicitHs() > 0 or anchor.GetNoImplicit():
        anchor.SetNumExplicitHs(anchor.GetNumExplicitHs() + int(bond_order))
    else:
        anchor.SetNoImplicit(False)
        anchor.SetNumExplicitHs(0)

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"remove_group({idx}, {root_idx}) produced an invalid molecule: {exc}"
        ) from exc

    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)
    return new_mol


# ===========================================================================
# Tests
# ===========================================================================
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

    class TestRemoveGroup(unittest.TestCase):
        # 1. Simple pendant: toluene -> benzene by removing the methyl.
        def test_toluene_to_benzene(self):
            m = _smi("Cc1ccccc1")  # 0=CH3, 1..6=ring
            out = remove_group(m, idx=1, root_idx=0)
            self.assertEqual(_canon(out), _canon(_smi("c1ccccc1")))
            self.assertEqual(out.GetNumAtoms(), 6)

        # 2. Larger pendant chain: remove OH from butanol -> butane.
        def test_butanol_remove_oh(self):
            m = _smi("CCCCO")  # 0..3=C, 4=O
            out = remove_group(m, idx=3, root_idx=4)
            self.assertEqual(_canon(out), _canon(_smi("CCCC")))

        # 3. Error: idx out of range.
        def test_error_idx_out_of_range(self):
            m = _smi("CC")
            with self.assertRaises(ValueError):
                remove_group(m, idx=99, root_idx=0)

        # 4. Error: root_idx out of range.
        def test_error_root_idx_out_of_range(self):
            m = _smi("CC")
            with self.assertRaises(ValueError):
                remove_group(m, idx=0, root_idx=99)

        # 5. Error: idx == root_idx.
        def test_error_same_index(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                remove_group(m, idx=0, root_idx=0)
            self.assertIn("distinct", str(cm.exception).lower())

        # 6. Error: no bond between idx and root_idx.
        def test_error_no_bond(self):
            m = _smi("CCC")
            with self.assertRaises(ValueError) as cm:
                remove_group(m, idx=0, root_idx=2)
            self.assertIn("no bond", str(cm.exception).lower())

        # 7. Error: bond is in a ring.
        def test_error_bond_in_ring(self):
            m = _smi("C1CCCCC1")
            with self.assertRaises(ValueError) as cm:
                remove_group(m, idx=0, root_idx=1)
            self.assertIn("ring", str(cm.exception).lower())

        # 8. Input mol unchanged.
        def test_input_mol_not_mutated(self):
            m = _smi("Cc1ccccc1")
            before = _canon(m)
            before_n = m.GetNumAtoms()
            _ = remove_group(m, idx=1, root_idx=0)
            self.assertEqual(_canon(m), before)
            self.assertEqual(m.GetNumAtoms(), before_n)

        # 9. Removed-group integrity + stereo cleared: 2-butene, remove a
        #    methyl that participates in the E/Z definition.
        def test_remove_methyl_clears_ez(self):
            m = _smi("C/C=C/C")  # 0=C,1=C,2=C,3=C
            out = remove_group(m, idx=1, root_idx=0)
            self.assertEqual(out.GetNumAtoms(), 3)
            self.assertEqual(_canon(out), _canon(_smi("C=CC")))
            self.assertNotIn("/", _canon(out))
            self.assertNotIn("\\", _canon(out))

        # 10. Stereo: chiral center loses a neighbor -> chiral tag cleared.
        def test_stereocenter_tag_cleared(self):
            m = _smi("[C@](F)(Cl)(Br)I")  # atom 0 is the stereocenter
            # Remove F (atom 1). Anchor 0 stays at index 0 (lower than 1).
            out = remove_group(m, idx=0, root_idx=1)
            self.assertEqual(
                out.GetAtomWithIdx(0).GetChiralTag(),
                Chem.ChiralType.CHI_UNSPECIFIED,
            )
            self.assertEqual(out.GetNumAtoms(), 4)

        # 11. Pendant with internal structure: strip the ethyl off ethyl
        #     acetate's ester oxygen -> acetic acid. Anchor on the -O- (3);
        #     root is the first ethyl C (4); fragment [4, 5] is removed.
        def test_remove_ethyl_from_ester(self):
            m = _smi("CC(=O)OCC")  # 0=C,1=C,2=O(=),3=O,4=C,5=C
            out = remove_group(m, idx=3, root_idx=4)
            self.assertEqual(_canon(out), _canon(_smi("CC(=O)O")))

        # 12. Pendant of size 1 and size N.
        def test_pendant_size_one(self):
            m = _smi("CO")  # methanol: remove OH -> methane
            out = remove_group(m, idx=0, root_idx=1)
            self.assertEqual(_canon(out), _canon(_smi("C")))
            self.assertEqual(out.GetNumAtoms(), 1)

        def test_pendant_size_large(self):
            # Phenyl ring + hexyl tail. Ring is atoms 0..5, tail atoms 6..11.
            m = _smi("c1ccccc1CCCCCC")
            self.assertIsNotNone(m.GetBondBetweenAtoms(5, 6))
            out = remove_group(m, idx=5, root_idx=6)
            self.assertEqual(_canon(out), _canon(_smi("c1ccccc1")))
            self.assertEqual(out.GetNumAtoms(), 6)

        # 13. Double-bond anchor: removing a =O refills valence correctly.
        def test_remove_double_bonded_oxygen(self):
            # Formaldehyde CH2=O: remove =O -> methane.
            m = _smi("C=O")
            out = remove_group(m, idx=0, root_idx=1)
            self.assertEqual(_canon(out), _canon(_smi("C")))

    unittest.main(verbosity=2)
