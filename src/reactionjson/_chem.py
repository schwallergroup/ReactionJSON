"""Small RDKit helpers shared by the route-mapping modules.

Everything here is keyed by *atom-map number*, never by atom index: indices
change every time an op edits a molecule (``add_group`` appends atoms,
``invert_stereocenter`` round-trips through canonical SMILES), while map numbers
are the one identity that survives a whole route.
"""
from __future__ import annotations

from typing import Iterable, Optional

import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

#: The route_ops renderer treats every map >= 900 as "unassigned" and renumbers
#: it (``_assign_offset_maps_inplace``), and add_group reserves 999 internally.
#: Global ids must therefore stay below this bound for a plan to render verbatim.
MAX_RENDERABLE_MAP = 899


def parse(smiles: str, *, sanitize: bool = True) -> Chem.Mol:
    """Parse SMILES or raise ``ValueError`` naming the offending string."""
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    return mol


def strip_maps(mol: Chem.Mol) -> Chem.Mol:
    """Copy of ``mol`` with every atom-map number cleared."""
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(0)
    return out


def canonical(smiles: str, *, keep_maps: bool = False, stereo: bool = True) -> str:
    """Canonical SMILES of ``smiles``, map-stripped unless ``keep_maps``.

    The stripped form is re-parsed once more: an atom that was written in
    brackets only because it carried a map (``[CH3:1]``) keeps an explicit-H
    flag after stripping, and a second parse normalises that away so two
    chemically identical strings compare equal.
    """
    mol = parse(smiles)
    if not keep_maps:
        mol = strip_maps(mol)
    smi = Chem.MolToSmiles(mol, isomericSmiles=stereo)
    if keep_maps:
        return smi
    again = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(again, isomericSmiles=stereo) if again is not None else smi


def canonical_mapped(smiles: str) -> str:
    """Canonical SMILES keeping maps: the exact string the renderer's frontier holds."""
    return Chem.MolToSmiles(parse(smiles))


def fragments(smiles: str) -> list[str]:
    """Mapped canonical SMILES of every connected component, sorted."""
    mol = parse(smiles)
    return sorted(Chem.MolToSmiles(f) for f in Chem.GetMolFrags(mol, asMols=True))


def fragment_multiset(smiles: str, *, keep_maps: bool = False, stereo: bool = True) -> list[str]:
    """Sorted canonical fragments: the comparison key for "same precursors"."""
    return sorted(canonical(f, keep_maps=keep_maps, stereo=stereo) for f in fragments(smiles))


def map_numbers(mol: Chem.Mol) -> list[int]:
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]


def atom_by_map(mol: Chem.Mol) -> dict[int, Chem.Atom]:
    """map -> atom for every mapped atom (map 0 atoms are skipped)."""
    return {a.GetAtomMapNum(): a for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}


def kekulized(mol: Chem.Mol) -> Chem.Mol:
    """Kekulé copy with aromatic flags cleared, exactly as the executor prepares it."""
    rw = Chem.RWMol(mol)
    Chem.Kekulize(rw, clearAromaticFlags=True)
    return rw.GetMol()


def kekule_orders(mol: Chem.Mol) -> dict[frozenset, int]:
    """{frozenset(map_a, map_b): integer bond order} for a Kekulé molecule."""
    out: dict[frozenset, int] = {}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        out[frozenset((a1, a2))] = int(round(b.GetBondTypeAsDouble()))
    return out


def kekulize_like(mol: Chem.Mol, reference: dict[frozenset, int]) -> Chem.Mol:
    """Kekulé form of ``mol`` whose double bonds agree with ``reference`` as far as possible.

    The bond-order diff between product and reactants is only meaningful if both
    sides use *compatible* Kekulé structures. The executor fixes the product's
    (it kekulizes the SMILES it is given); an independently kekulized reactant
    may alternate the other way round an untouched benzene ring, and a naive
    diff would then report six spurious bond-order changes on a ring the
    reaction never touched.

    RDKit's own kekulization tells us which aromatic atoms need a pi bond; any
    perfect matching of those atoms over aromatic bonds is an equally valid
    Kekulé structure. We pick the matching that maximises agreement with the
    reference (max-weight, max-cardinality matching: every edge weighs 1, an
    edge that is double in the reference weighs 2).
    """
    kek = kekulized(mol)
    aromatic_bonds = [b for b in mol.GetBonds() if b.GetIsAromatic()]
    if not aromatic_bonds:
        return kek
    needs_pi: set[int] = set()
    for b in aromatic_bonds:
        kb = kek.GetBondWithIdx(b.GetIdx())
        if kb.GetBondType() == Chem.BondType.DOUBLE:
            needs_pi.update((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    graph = nx.Graph()
    for b in aromatic_bonds:
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in needs_pi and j in needs_pi:
            key = frozenset((mol.GetAtomWithIdx(i).GetAtomMapNum(),
                             mol.GetAtomWithIdx(j).GetAtomMapNum()))
            graph.add_edge(i, j, weight=2 if reference.get(key) == 2 else 1)
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    if 2 * len(matching) != len(needs_pi):
        return kek  # cannot happen for a kekulizable mol; keep RDKit's choice
    doubles = {frozenset(e) for e in matching}
    rw = Chem.RWMol(kek)
    for b in aromatic_bonds:
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        rb = rw.GetBondBetweenAtoms(i, j)
        rb.SetBondType(Chem.BondType.DOUBLE if frozenset((i, j)) in doubles
                       else Chem.BondType.SINGLE)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def stereo_labels(mol: Chem.Mol) -> tuple[dict[int, str], dict[frozenset, str]]:
    """Map-resolved stereo descriptors: ({map: 'R'|'S'}, {bond: 'E'|'Z'}).

    Chiral tags are relative to each atom's neighbour *order*, which ops shuffle,
    so tags from two molecules cannot be compared directly. CIP labels can,
    provided both molecules have the same constitution. Labelling every atom's
    isotope with its map number makes each mapped atom unique, so the descriptor
    is well defined even at centres whose substituents differ only by map
    (the descriptor is then consistent, not chemically meaningful — which is all
    a same-constitution comparison needs).
    """
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetIsotope(a.GetAtomMapNum())
    try:
        Chem.AssignStereochemistry(m, cleanIt=False, force=True)
        rdCIPLabeler.AssignCIPLabels(m)
    except (RuntimeError, ValueError):
        return {}, {}
    atoms: dict[int, str] = {}
    for a in m.GetAtoms():
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and a.HasProp("_CIPCode"):
            code = a.GetProp("_CIPCode")
            if code in ("R", "S", "r", "s"):
                atoms[a.GetAtomMapNum()] = code.upper()
    bonds: dict[frozenset, str] = {}
    for b in m.GetBonds():
        if b.HasProp("_CIPCode") and b.GetStereo() != Chem.BondStereo.STEREONONE:
            code = b.GetProp("_CIPCode")
            if code in ("E", "Z"):
                key = frozenset((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))
                bonds[key] = code
    return atoms, bonds


def has_tag(mol: Chem.Mol, map_num: int) -> bool:
    atom = atom_by_map(mol).get(map_num)
    return atom is not None and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED


def symmetry_classes(mol: Chem.Mol) -> dict[int, int]:
    """map -> canonical symmetry class (ties unbroken), computed map-free.

    Two atoms in the same class are interchangeable by an automorphism, so a
    mapping that differs from ground truth only by swapping them is equivalent.
    """
    ranks = list(Chem.CanonicalRankAtoms(strip_maps(mol), breakTies=False))
    return {a.GetAtomMapNum(): ranks[a.GetIdx()] for a in mol.GetAtoms()}


def max_map(smiles_list: Iterable[str]) -> int:
    best = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            best = max([best] + map_numbers(mol))
    return best


def split_reaction(rxn: str) -> tuple[str, str]:
    """``left>>right`` -> (left, right); agents (``a>b>c``) are folded into the left side."""
    if ">>" in rxn:
        left, right = rxn.split(">>", 1)
        return left, right
    parts = rxn.split(">")
    if len(parts) == 3:
        left = ".".join(p for p in (parts[0], parts[1]) if p)
        return left, parts[2]
    raise ValueError(f"not a reaction SMILES: {rxn!r}")


def safe_canonical(smiles: str) -> Optional[str]:
    try:
        return canonical(smiles)
    except ValueError:
        return None
