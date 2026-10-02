"""Atomic op: set_formal_charge.

Sets the formal charge on a single atom and refills its hydrogens to the
default valence for the new charge, the same way break_bond refills freed
valence. So a protonation-state change is one op (R-COO- -> R-COOH,
R-NH3+ -> R-NH2), and a charge that comes with new bonds (Ar-NH2 -> Ar-NO2)
leaves H that the following add_group / add_bond consume. A count the refill
does not get right is set with set_explicit_h afterwards.
"""

from rdkit import Chem

from reactionjson.ops._sanitize import sanitize as _sanitize


def set_formal_charge(mol: Chem.Mol, idx: int, charge: int) -> Chem.Mol:
    """Set the formal charge of atom idx to ``charge`` and refill its H count.

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - idx is out of range
        - charge is not an integer in [-4, 4]
        - the resulting molecule fails sanitization (valence violation)
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")
    if isinstance(charge, bool) or not isinstance(charge, int) or not -4 <= charge <= 4:
        raise ValueError(f"charge must be an integer in [-4, 4]; got {charge!r}")

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)
    atom.SetFormalCharge(charge)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)
    atom.SetNumRadicalElectrons(0)

    new_mol = rw.GetMol()
    try:
        _sanitize(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"set_formal_charge({idx}, {charge}) produced an invalid molecule: {exc}"
        ) from exc

    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)
    return new_mol
