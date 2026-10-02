"""Sanitization shared by every atomic op.

``Chem.SanitizeMol`` re-perceives aromaticity, which silently undoes a
kekulization. v2's contract is that the executor kekulizes **once**, before the
first op, and the model edits the Kekule structure it was shown; if each op
re-aromatized, the second op onward would meet an aromatic molecule again, and
re-kekulizing per op can land on a *different* resonance structure than the one
the model saw — so a bond it read as double may come back single.

So while a kekulized sequence is running, ops sanitize with aromaticity
perception switched off and the Kekule structure persists untouched. Aromaticity
is restored once, by the executor's finalize, so Kekule form never leaks into
the returned SMILES.
"""

from __future__ import annotations

from contextlib import contextmanager

from rdkit import Chem

_KEKULE_ACTIVE = False


@contextmanager
def kekule_sequence():
    """Mark the enclosing op sequence as running on a kekulized molecule."""
    global _KEKULE_ACTIVE
    prev = _KEKULE_ACTIVE
    _KEKULE_ACTIVE = True
    try:
        yield
    finally:
        _KEKULE_ACTIVE = prev


def sanitize(mol) -> None:
    """SanitizeMol that preserves an in-force Kekule structure."""
    if _KEKULE_ACTIVE:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    else:
        Chem.SanitizeMol(mol)
