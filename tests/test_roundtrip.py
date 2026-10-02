"""apply_ops and derive_ops must be inverse, and the ladder must behave."""
import pytest
from rdkit import Chem, RDLogger

from reactionjson import OPS, apply_ops, derive_ops, get_mapped_smiles

RDLogger.DisableLog("rdApp.*")


def canon_set(smiles):
    mol = Chem.MolFromSmiles(smiles)
    out = []
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
        f = Chem.Mol(frag)
        for a in f.GetAtoms():
            a.SetAtomMapNum(0)
        out.append(Chem.MolToSmiles(f))
    return frozenset(out)


def test_vocabulary_is_nine_ops():
    assert len(OPS) == 9
    assert "change_atom" not in OPS       # retired: not a reaction
    assert "change_bond_order" not in OPS  # retired: subsumed by the ladder
    assert "remove_group" not in OPS       # retired: discarded a precursor


def test_break_bond_steps_down_one_order():
    """A carbonyl loses its pi bond first — that is a retro-reduction."""
    # CC=O maps as [CH3:1][CH:2]=[O:3], so the carbonyl is the 2-3 bond.
    out, err = apply_ops([{"op": "break_bond", "map_a": 2, "map_b": 3}],
                         get_mapped_smiles("CC=O"))
    assert err is None
    assert canon_set(out) == canon_set("CCO")


def test_aromatic_ring_opens():
    """Kekule form makes a ring bond editable like any other."""
    out, err = apply_ops([{"op": "break_bond", "map_a": 3, "map_b": 4}],
                         get_mapped_smiles("c1ccncc1"))
    assert err is None
    assert Chem.MolFromSmiles(out) is not None


def test_second_ring_still_editable_after_the_first_op():
    """The single kekulization must survive the whole sequence."""
    out, err = apply_ops(
        [{"op": "break_bond", "map_a": 4, "map_b": 5},
         {"op": "break_bond", "map_a": 1, "map_b": 2}],
        get_mapped_smiles("c1ccc2[nH]ccc2c1"))
    assert err is None, err
    assert Chem.MolFromSmiles(out).GetNumHeavyAtoms() == 9


def test_kekule_does_not_leak_into_the_result():
    """A ring that survives comes back aromatic, not as alternating bonds."""
    out, err = apply_ops([], get_mapped_smiles("c1ccncc1"))
    assert err is None
    assert "=" not in out


def test_errors_name_the_failing_op_and_do_not_raise():
    out, err = apply_ops([{"op": "add_group", "map_idx": 1, "fragment_smiles": "O"}],
                         get_mapped_smiles("CCO"))
    assert out is None
    assert "Op [0]" in err and "*" in err


def test_retired_ops_are_rejected_by_name():
    out, err = apply_ops([{"op": "change_atom", "map_idx": 1, "symbol": "N"}],
                         get_mapped_smiles("c1ccccc1"))
    assert out is None
    assert "not in vocabulary v2" in err


@pytest.mark.parametrize("reaction", [
    # retro-esterification: break the C-O, both halves are precursors
    "[CH3:1][C:2](=[O:3])[O:4][CH3:5]>>[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][OH:6]",
    # retro nitro reduction: a charge change, no NH3 artifact
    "[NH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1>>[O-][N+:1](=O)[c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1",
    # carboxylate -> acid: one set_formal_charge, H refilled
    "[CH3:1][C:2](=[O:3])[O-:4]>>[CH3:1][C:2](=[O:3])[OH:4]",
    # quaternisation undone: R4N+ -> R3N + MeI
    "[CH3:1][N+:2]([CH3:3])([CH3:4])[CH3:5]>>[CH3:1][N:2]([CH3:3])[CH3:4].[CH3:5]I",
    # product drawn without stereo, precursor has it: set_stereocenter
    "[CH3:1][CH:2]([NH2:3])[C:4](=[O:5])[O:6][CH3:7]>>[CH3:1][C@H:2]([NH2:3])[C:4](=[O:5])[OH:6].[CH3:7]O",
    # a severed bond at a stereocentre clears its tag; set_stereocenter restores it
    "[CH3:1][C@H:2]([NH:3][CH3:8])[CH2:4][CH3:5]>>[CH3:1][C@@H:2]([NH2:3])[CH2:4][CH3:5].[CH3:8]I",
])
def test_derive_then_apply_round_trips(reaction):
    product, reactants = reaction.split(">>")
    ops = derive_ops(reaction)
    out, err = apply_ops(ops, product)
    assert err is None, err
    assert canon_set(out) == canon_set(reactants)


def test_set_formal_charge_refills_hydrogens():
    out, err = apply_ops([{"op": "set_formal_charge", "map_idx": 3, "charge": 0}],
                         "[CH3:1][C:2](=[O:4])[O-:3]")
    assert err is None, err
    assert canon_set(out) == canon_set("CC(=O)O")


def test_set_stereocenter_sets_absolute_configuration():
    out, err = apply_ops([{"op": "set_stereocenter", "map_idx": 2, "stereo": "S"}],
                         "[CH3:1][CH:2]([NH2:3])[C:4](=[O:5])[OH:6]")
    assert err is None, err
    assert canon_set(out) == canon_set("C[C@H](N)C(=O)O")  # L-alanine is S


def test_set_stereocenter_refuses_a_non_centre():
    out, err = apply_ops([{"op": "set_stereocenter", "map_idx": 2, "stereo": "R"}],
                         "[CH3:1][CH2:2][OH:3]")
    assert out is None and "not a stereocentre" in err


def test_set_bond_stereo_with_reversed_endpoints():
    """Map order opposite to the bond's begin/end must not crash."""
    smi = "[CH3:1][CH:2]=[CH:3][CH3:4]"
    for a, b in ((2, 3), (3, 2)):
        out, err = apply_ops([{"op": "set_bond_stereo", "map_a": a, "map_b": b, "stereo": "E"}], smi)
        assert err is None, err
        assert canon_set(out) == canon_set("C/C=C/C")
