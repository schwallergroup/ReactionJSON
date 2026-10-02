"""Repair atom mappings that assert an element transmutation.

v1's ``change_atom`` let a route claim one atom *becomes* another element — a
deoxybromination mapped as "map 32 is Br in the product and O in the precursor".
The BE diff rejects that, correctly: it does not conserve atoms, and deriving
ops from it would invent chemistry.

The chemistry underneath is a substituent swap, and that *is* expressible. This
rewrites the mapping to say so: the product's atom leaves (it reappears in the
reactants as its own fragment) and the reactant's atom is new (it gets a fresh
map, so the derivation introduces it with ``add_group``).

    product   Ar-[Br:32]
    reactants Ar-[OH:32]                    <- rejected, Br is not O
    repaired  Ar-[OH:99] . [BrH:32]         <- break C-Br, add_group *O
"""

from __future__ import annotations

from rdkit import Chem

__all__ = ["repair_element_mismatches", "repair_absent_atoms", "repair_mapping", "RepairError"]


class RepairError(ValueError):
    """The mapping could not be repaired."""


def _atoms_by_map(mol):
    return {a.GetAtomMapNum(): a for a in mol.GetAtoms() if a.GetAtomMapNum()}


def repair_element_mismatches(mapped_reaction: str) -> tuple[str, list[str]]:
    """Rewrite element-mismatched maps as a leaving group plus an incoming atom.

    :param mapped_reaction: ``"<mapped_product>>><mapped_reactants>"``
    :returns: ``(repaired_reaction, notes)``. ``notes`` is empty when nothing
        needed repairing, and the reaction string comes back unchanged.
    :raises RepairError: when the reaction cannot be parsed.
    """
    product_smi, reactants_smi = mapped_reaction.split(">>", 1)
    product = Chem.MolFromSmiles(product_smi, sanitize=False)
    reactants = Chem.MolFromSmiles(reactants_smi, sanitize=False)
    if product is None or reactants is None:
        raise RepairError("could not parse the mapped reaction")

    p_atoms, r_atoms = _atoms_by_map(product), _atoms_by_map(reactants)
    used = set(p_atoms) | set(r_atoms)
    nxt = max(used) + 1 if used else 1

    notes, leaving = [], []
    for m, pa in sorted(p_atoms.items()):
        ra = r_atoms.get(m)
        if ra is None or ra.GetAtomicNum() == pa.GetAtomicNum():
            continue
        # The reactant's atom is genuinely new -> fresh map, so it is introduced.
        ra.SetAtomMapNum(nxt)
        notes.append(
            f"map {m}: product {pa.GetSymbol()} leaves; "
            f"reactant {ra.GetSymbol()} is new (map {nxt})"
        )
        nxt += 1
        # The product's atom has to reappear among the reactants, or the diff
        # still does not conserve atoms. It leaves as its own neutral fragment.
        leaving.append((pa.GetSymbol(), m))

    if not notes:
        return mapped_reaction, []

    frags = [Chem.MolToSmiles(reactants)]
    for symbol, m in leaving:
        rw = Chem.RWMol()
        atom = Chem.Atom(symbol)
        atom.SetAtomMapNum(m)
        rw.AddAtom(atom)
        frag = rw.GetMol()
        try:
            Chem.SanitizeMol(frag)
        except Exception as exc:  # pragma: no cover - defensive
            raise RepairError(f"could not build the leaving fragment {symbol}: {exc}")
        frags.append(Chem.MolToSmiles(frag))

    return f"{product_smi}>>{'.'.join(frags)}", notes


def repair_absent_atoms(mapped_reaction: str) -> tuple[str, list[str]]:
    """Give back product atoms that the reaction dropped.

    v1's ``remove_group`` detached a group and discarded it, so the recorded
    reactants are missing atoms the product had. v2's position is that nothing
    is deleted — severing a bond yields two fragments and both are precursors —
    so the repair is to put the orphaned atoms back on the reactant side as
    their own fragment.

    The orphans are lifted out as connected fragments of the *product*, not one
    atom at a time, so a discarded protecting group comes back whole with its
    internal bonds and its map numbers intact.

        product   Ar-[CH2:7][c:8]1...(a benzyl group)
        reactants Ar-H                      <- maps 7, 8, ... absent
        repaired  Ar-H . [CH3:7][c:8]1...   <- break the bond, both halves kept
    """
    product_smi, reactants_smi = mapped_reaction.split(">>", 1)
    product = Chem.MolFromSmiles(product_smi, sanitize=False)
    reactants = Chem.MolFromSmiles(reactants_smi, sanitize=False)
    if product is None or reactants is None:
        raise RepairError("could not parse the mapped reaction")

    r_maps = {a.GetAtomMapNum() for a in reactants.GetAtoms() if a.GetAtomMapNum()}
    orphans = [a.GetIdx() for a in product.GetAtoms()
               if a.GetAtomMapNum() and a.GetAtomMapNum() not in r_maps]
    if not orphans:
        return mapped_reaction, []

    # Keep only the orphaned atoms, so the connected components among them are
    # the discarded groups. Bonds to atoms that survived are the ones that broke.
    rw = Chem.RWMol(product)
    for idx in sorted((a.GetIdx() for a in product.GetAtoms()), reverse=True):
        if idx not in orphans:
            rw.RemoveAtom(idx)
    # The atoms carry the hydrogen counts they had *bonded to the product*. Left
    # as-is, a benzyl that was attached through its CH2 comes back as the carbene
    # [CH2]c1ccccc1 instead of toluene. Clear the counts and let RDKit refill
    # valence, so the severed fragment is the species that actually leaves.
    for atom in rw.GetAtoms():
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    try:
        sub = rw.GetMol()
        Chem.SanitizeMol(sub)
    except Exception as exc:
        raise RepairError(f"could not lift the discarded atoms out: {exc}")

    pieces = [Chem.MolToSmiles(f) for f in
              Chem.GetMolFrags(sub, asMols=True, sanitizeFrags=True)]
    maps = sorted(product.GetAtomWithIdx(i).GetAtomMapNum() for i in orphans)
    notes = [f"maps {maps} were dropped by the reaction; "
             f"returned as {len(pieces)} precursor fragment(s)"]
    return f"{product_smi}>>{'.'.join([Chem.MolToSmiles(reactants)] + pieces)}", notes


def repair_mapping(mapped_reaction: str) -> tuple[str, list[str]]:
    """Both repairs, in the order the diff needs them.

    Element mismatches first: that repair *adds* the leaving atom to the
    reactants, which would otherwise look like one more absent atom.
    """
    out, notes = repair_element_mismatches(mapped_reaction)
    out, more = repair_absent_atoms(out)
    return out, notes + more
