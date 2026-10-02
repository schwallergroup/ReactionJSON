"""Atomic op: set_stereocenter.

Gives a tetrahedral centre an absolute configuration, R or S (CIP), whether or
not it already carries a chiral tag. This is the one op that *creates* a tag:
break_bond clears the tag at both endpoints, and a product drawn without stereo
can lead to a precursor that has it, so invert_stereocenter alone cannot reach
every precursor.
"""

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from reactionjson.ops._sanitize import sanitize as _sanitize

_TAGS = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _cip(mol: Chem.Mol, idx: int):
    m = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(m)
    atom = m.GetAtomWithIdx(idx)
    return atom.GetProp("_CIPCode").upper() if atom.HasProp("_CIPCode") else None


def set_stereocenter(mol: Chem.Mol, idx: int, stereo: str) -> Chem.Mol:
    """Set atom idx to CIP configuration ``stereo`` ('R' or 'S', case-insensitive).

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - idx is out of range
        - stereo is not 'R' or 'S'
        - the atom is not a CIP stereocentre (neither tag gives it a label)
    """
    want = stereo.upper() if isinstance(stereo, str) else None
    if want not in ("R", "S"):
        raise ValueError(f"stereo must be 'R' or 'S'; got {stereo!r}")
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")

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
