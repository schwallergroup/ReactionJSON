"""Atomic op: set_stereocenter.

Gives a tetrahedral centre an absolute configuration, R or S (CIP), whether or
not it already carries a chiral tag, or clears it with ``stereo=None``. The one
tetrahedral stereo op: inverting is setting the opposite label, and it can
create a tag, which break_bond (clears both endpoints) and products drawn
without stereo both require.
"""

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from reactionjson.ops._sanitize import sanitize as _sanitize

_TAGS = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _cip(mol: Chem.Mol, idx: int):
    # Label the molecule as it will be written, not the live one: after a
    # sequence of edits the in-memory tag can read as R while the SMILES it
    # writes re-parses as S (seen on a ring CH whose bond list was rebuilt by
    # break_bond + add_group).
    smi = Chem.MolToSmiles(mol)
    order = list(mol.GetProp("_smilesAtomOutputOrder", autoConvert=True))
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    rdCIPLabeler.AssignCIPLabels(m)
    atom = m.GetAtomWithIdx(order.index(idx))
    return atom.GetProp("_CIPCode").upper() if atom.HasProp("_CIPCode") else None


def set_stereocenter(mol: Chem.Mol, idx: int, stereo) -> Chem.Mol:
    """Set atom idx to CIP configuration ``stereo`` ('R'/'S', case-insensitive; None clears).

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - idx is out of range
        - stereo is not 'R', 'S' or None
        - the atom is not a CIP stereocentre (neither tag gives it a label)
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")
    if stereo is None:
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        new_mol = rw.GetMol()
        _sanitize(new_mol)
        Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)
        return new_mol
    want = stereo.upper() if isinstance(stereo, str) else None
    if want not in ("R", "S"):
        raise ValueError(f"stereo must be 'R', 'S' or null; got {stereo!r}")

    for tag in _TAGS:
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(idx).SetChiralTag(tag)
        new_mol = rw.GetMol()
        try:
            _sanitize(new_mol)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
            raise ValueError(
                f"set_stereocenter({idx}, {stereo!r}) produced an invalid molecule: {exc}"
            ) from exc
        Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)
        label = _cip(new_mol, idx)
        if label is None:
            break
        if label == want:
            return new_mol
    raise ValueError(f"atom {idx} is not a stereocentre; it has no R/S configuration to set")
