"""Atomic op: add_bond.

Adds bond order between two atoms, returning a new sanitized Mol. Implicit
hydrogens on the two endpoints are decremented as needed so the new bond does
not create a valence violation.

Semantics (v2): add_bond is the inverse ladder to break_bond — nothing -> single
-> double -> triple. When no bond exists it creates one of the requested
`order`; when a bond already exists it RAISES the order by `order` (default 1).
That subsumes `change_bond_order` (delta=+n is n add_bond calls) and is strictly
more permissive than the original op, which refused when a bond was already
present — so no sequence that worked before can change behaviour.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


_ORDER_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

_ORDER_DELTA = {1: 1, 2: 2, 3: 3}


def add_bond(mol: Chem.Mol, idx_a: int, idx_b: int, order: int = 1) -> Chem.Mol:
    """Add a bond between atoms idx_a and idx_b of order 1, 2, or 3.

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - order is not in {1, 2, 3}
        - idx_a == idx_b
        - either index is out of range
        - a bond already exists between the two atoms
        - the resulting molecule fails sanitization (e.g. valence violation)
    """
    if order not in _ORDER_MAP:
        raise ValueError(f"order must be 1, 2, or 3; got {order!r}")
    if idx_a == idx_b:
        raise ValueError(f"idx_a and idx_b must differ (got {idx_a})")

    n = mol.GetNumAtoms()
    if not (0 <= idx_a < n) or not (0 <= idx_b < n):
        raise ValueError(
            f"atom index out of range: idx_a={idx_a}, idx_b={idx_b}, num_atoms={n}"
        )

    # A bond is already there: raise its order instead of refusing. Delegated so
    # the implicit-H and E/Z bookkeeping lives in exactly one place.
    if mol.GetBondBetweenAtoms(idx_a, idx_b) is not None:
        from reactionjson.ops.change_bond_order import change_bond_order
        return change_bond_order(mol, idx_a, idx_b, _ORDER_DELTA[order])

    rw = Chem.RWMol(mol)
    delta = _ORDER_DELTA[order]

    # RDKit will NOT auto-reduce implicit Hs when you call AddBond on atoms that
    # have "noImplicit" semantics or when explicit Hs are set. To be robust, we
    # explicitly decrement either numExplicitHs (if set) or rely on the implicit
    # H calculation by updating the property cache after the edit. In practice:
    # if an atom was built by SMILES parsing (e.g. "C"), it has numExplicitHs=0
    # and its H count is implicit-from-valence. Decrementing explicit Hs when
    # they are already zero would fail, so we only touch explicit Hs when set.
    # Only give up hydrogens where the atom has no spare valence. A fragment
    # introduced by add_fragment is written as a GROUP with an open valence
    # (e.g. "[CH3:101]" is a methyl to be attached, not methane), so
    # decrementing its H unconditionally would silently produce a CH2. Atoms
    # that are already saturated do lose an H, which is what makes bonding onto
    # an alcohol oxygen or an amine nitrogen work.
    pt = Chem.GetPeriodicTable()
    for idx in (idx_a, idx_b):
        atom = rw.GetAtomWithIdx(idx)
        used = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        try:
            default_valence = pt.GetDefaultValence(atom.GetAtomicNum())
        except RuntimeError:
            default_valence = -1
        spare = (default_valence - int(used) - atom.GetTotalNumHs()
                 + abs(atom.GetFormalCharge()) * 0)
        if default_valence < 0:
            spare = delta          # unknown valence: leave H alone
        if spare >= delta:
            continue
        needed = delta - max(spare, 0)
        n_expl = atom.GetNumExplicitHs()
        if n_expl >= needed:
            atom.SetNumExplicitHs(n_expl - needed)
        elif n_expl > 0:
            atom.SetNumExplicitHs(0)
        # else: rely on implicit-H recompute during sanitization.

    rw.AddBond(idx_a, idx_b, _ORDER_MAP[order])

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"add_bond({idx_a}, {idx_b}, order={order}) produced an invalid "
            f"molecule: {exc}"
        ) from exc

    # SanitizeMol does not re-perceive stereochemistry; the new bond may
    # create or destroy a stereocenter elsewhere, so force a fresh pass.
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

    class TestAddBond(unittest.TestCase):
        # 1. Simple: C.C -> CC
        def test_simple_single_bond_joins_two_methanes(self):
            m = _smi("C.C")
            out = add_bond(m, 0, 1, order=1)
            self.assertEqual(_canon(out), _canon(_smi("CC")))

        # 2. Double bond that works: ethane -> ethene is not valid (too many Hs),
        #    but two methyls with explicit-radical-like setup is awkward; instead
        #    add a double bond between two CH3 fragments: C.C with order=2
        #    => H2C=CH2 (ethene).
        def test_double_bond_between_two_methyl_fragments(self):
            m = _smi("C.C")
            out = add_bond(m, 0, 1, order=2)
            self.assertEqual(_canon(out), _canon(_smi("C=C")))

        # 2b. Triple bond: C.C -> C#C (acetylene)
        def test_triple_bond_between_two_methyl_fragments(self):
            m = _smi("C.C")
            out = add_bond(m, 0, 1, order=3)
            self.assertEqual(_canon(out), _canon(_smi("C#C")))

        # 3. Error: bond already exists
        def test_error_bond_already_exists(self):
            m = _smi("CC")
            with self.assertRaises(ValueError) as cm:
                add_bond(m, 0, 1, order=1)
            self.assertIn("already exists", str(cm.exception))

        # 4. Error: out-of-range index
        def test_error_index_out_of_range(self):
            m = _smi("CC")
            with self.assertRaises(ValueError):
                add_bond(m, 0, 99, order=1)
            with self.assertRaises(ValueError):
                add_bond(m, -1, 1, order=1)

        # 5. Error: order = 4
        def test_error_order_four(self):
            m = _smi("C.C")
            with self.assertRaises(ValueError) as cm:
                add_bond(m, 0, 1, order=4)
            self.assertIn("order", str(cm.exception))

        # 6. Error: valence violation (add bond to fully saturated carbon).
        #    In C.C, each C has 4 implicit Hs. Adding ONE bond works (becomes
        #    ethane). Adding TWO bonds between same atoms is caught earlier as
        #    "already exists". To actually force a valence violation, we try to
        #    add a bond to a carbon that already has 4 bonds: CH4 has 0 heavy
        #    neighbors but 4 Hs; in neopentane C(C)(C)(C)C the central C has 4
        #    heavy bonds already. Adding a 5th bond between the central C and a
        #    new atom is impossible (no spare atom in-mol), so use methane added
        #    to neopentane: atom 0 in "CC(C)(C)C" is a terminal CH3; atom 1 is
        #    the quaternary C with 4 bonds. Add a bond between atoms 0 and 1?
        #    already exists. So use two fragments: "C.CC(C)(C)C" — atom 0 is
        #    isolated CH4; atom 2 is the quaternary carbon (4 heavy bonds).
        def test_error_valence_violation_on_saturated_carbon(self):
            m = _smi("C.CC(C)(C)C")
            # indices: 0=CH4, 1=CH3, 2=C(quaternary), 3=CH3, 4=CH3, 5=CH3
            self.assertEqual(m.GetAtomWithIdx(2).GetDegree(), 4)
            with self.assertRaises(ValueError) as cm:
                add_bond(m, 0, 2, order=1)
            self.assertIn("invalid", str(cm.exception).lower())

        # 7. Ring-forming: chain -> ring, ring count increases by 1
        def test_ring_forming_increases_ring_count(self):
            # Hexane CCCCCC -> cyclohexane
            m = _smi("CCCCCC")
            ri_before = m.GetRingInfo().NumRings()
            out = add_bond(m, 0, 5, order=1)
            ri_after = out.GetRingInfo().NumRings()
            self.assertEqual(ri_after, ri_before + 1)
            self.assertEqual(_canon(out), _canon(_smi("C1CCCCC1")))

        # 8. Input mol unchanged
        def test_input_mol_not_mutated(self):
            m = _smi("C.C")
            before = _canon(m)
            before_n_bonds = m.GetNumBonds()
            _ = add_bond(m, 0, 1, order=1)
            self.assertEqual(_canon(m), before)
            self.assertEqual(m.GetNumBonds(), before_n_bonds)

        # Extra: same index for both endpoints.
        def test_error_same_index(self):
            m = _smi("CC")
            with self.assertRaises(ValueError):
                add_bond(m, 0, 0, order=1)

    unittest.main(verbosity=2)
