"""Bond–electron (BE) matrices of a mapped reaction and their difference.

Dugundji–Ugi: a molecule (or ensemble of molecules) over atoms 1..n is the
symmetric matrix B with B[i, j] = formal bond order between atoms i and j and
B[i, i] = the number of non-bonding valence electrons on atom i. For a mapped
reaction the reaction matrix R = B(reactants) − B(product), taken over the union
of atom maps, records exactly which bonds are made, broken or change order and
where electrons move. Everything else in the route-mapping package reads the
reaction off this diff rather than off SMILES strings.

Two choices make the diff usable for op generation:

* **Kekulé, not aromatic.** The v2 op vocabulary edits integer bond orders
  (``break_bond``/``add_bond`` step one order at a time), and the executor
  kekulizes the product before every op. The product side therefore uses the
  executor's own Kekulé structure; the reactant side is kekulized *to agree
  with it* (:func:`~reactionjson._chem.kekulize_like`) so an untouched
  aromatic ring contributes no spurious changes.
* **H counts, charges and stereo alongside the matrix.** Hydrogens are implicit
  in SMILES, so they are tracked as a per-atom vector rather than as matrix
  rows; formal charge and stereo descriptors likewise. The matrix diagonal is
  derived from them (outer-shell electrons − charge − bonds − H).

Direction: this module speaks *retro*. "product" is the molecule the step
starts from, "reactants" are what the ops must produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem

from reactionjson._chem import (
    atom_by_map,
    kekule_orders,
    kekulize_like,
    kekulized,
    parse,
    stereo_labels,
)

_PT = Chem.GetPeriodicTable()


@dataclass(frozen=True)
class AtomRecord:
    """Everything about one atom the op vocabulary can (or cannot) change."""

    map: int
    symbol: str
    charge: int
    hs: int
    isotope: int
    stereo: Optional[str]  # map-resolved CIP descriptor 'R'/'S', None if unassigned
    has_tag: bool          # a chiral tag is present (even when no descriptor is)
    fragment: int          # which connected component (molecule) the atom is in


@dataclass
class BEMatrix:
    """Kekulé bond orders + per-atom state for a set of mapped molecules."""

    atoms: dict[int, AtomRecord]
    bonds: dict[frozenset, int]
    bond_stereo: dict[frozenset, str]
    mol: Chem.Mol = field(repr=False)

    def order(self, a: int, b: int) -> int:
        return self.bonds.get(frozenset((a, b)), 0)

    def neighbours(self, a: int) -> set[int]:
        return {m for pair in self.bonds for m in pair if a in pair and m != a}

    def matrix(self, maps: list[int]) -> np.ndarray:
        """The Dugundji–Ugi BE matrix over ``maps`` (atoms absent here are zero rows).

        Off-diagonal: Kekulé bond order. Diagonal: non-bonding valence electrons,
        i.e. outer-shell electrons − formal charge − Σ bond orders − H count.
        """
        idx = {m: i for i, m in enumerate(maps)}
        mat = np.zeros((len(maps), len(maps)), dtype=int)
        for pair, order in self.bonds.items():
            a, b = tuple(pair)
            if a in idx and b in idx:
                mat[idx[a], idx[b]] = mat[idx[b], idx[a]] = order
        for m, rec in self.atoms.items():
            if m not in idx:
                continue
            bonded = sum(o for p, o in self.bonds.items() if m in p)
            outer = _PT.GetNOuterElecs(rec.symbol)
            mat[idx[m], idx[m]] = outer - rec.charge - bonded - rec.hs
        return mat


def be_matrix(mol: Chem.Mol, reference: Optional[dict[frozenset, int]] = None,
              *, kekulize: bool = True) -> BEMatrix:
    """Build the BE representation of a (sanitized, fully mapped) molecule.

    ``reference`` steers the Kekulé choice (see :func:`kekulize_like`); without
    it RDKit's kekulization is used, which is what the executor sees for a
    product parsed from SMILES.
    """
    if kekulize:
        kek = kekulize_like(mol, reference) if reference is not None else kekulized(mol)
    else:
        kek = mol
    labels, bond_labels = stereo_labels(mol)
    frag_of: dict[int, int] = {}
    for fi, idxs in enumerate(Chem.GetMolFrags(mol)):
        for i in idxs:
            frag_of[i] = fi
    atoms: dict[int, AtomRecord] = {}
    for a in mol.GetAtoms():
        m = a.GetAtomMapNum()
        atoms[m] = AtomRecord(
            map=m,
            symbol=a.GetSymbol(),
            charge=a.GetFormalCharge(),
            hs=a.GetTotalNumHs(),
            isotope=a.GetIsotope(),
            stereo=labels.get(m),
            has_tag=a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED,
            fragment=frag_of[a.GetIdx()],
        )
    return BEMatrix(atoms=atoms, bonds=kekule_orders(kek), bond_stereo=bond_labels, mol=kek)


@dataclass(frozen=True)
class BondChange:
    """One bond whose Kekulé order differs; ``product``→``reactant`` is the retro sense."""

    a: int
    b: int
    product: int
    reactant: int

    @property
    def kind(self) -> str:
        if self.product == 0:
            return "formed"      # present only in the reactants (retro: formed)
        if self.reactant == 0:
            return "broken"      # present only in the product (retro: broken)
        return "order"


@dataclass
class ReactionDiff:
    """R = B(reactants) − B(product), plus the per-atom changes the matrix omits."""

    product: BEMatrix
    reactants: BEMatrix
    product_maps: set[int]
    reactant_only_maps: set[int]
    bond_changes: list[BondChange]
    h_changes: dict[int, tuple[int, int]]
    charge_changes: dict[int, tuple[int, int]]
    stereo_changes: dict[int, tuple[Optional[str], Optional[str]]]
    bond_stereo_changes: dict[frozenset, tuple[Optional[str], Optional[str]]]

    def reaction_matrix(self) -> tuple[list[int], np.ndarray]:
        maps = sorted(self.product_maps | self.reactant_only_maps)
        return maps, self.reactants.matrix(maps) - self.product.matrix(maps)

    @property
    def reaction_centre(self) -> set[int]:
        """Product atoms whose bonding, H count, charge or stereo changes.

        An atom that gains a bond to a reactant-only atom (a leaving group in the
        forward sense) is included through the bond change itself.
        """
        centre: set[int] = set()
        for bc in self.bond_changes:
            centre.update(m for m in (bc.a, bc.b) if m in self.product_maps)
        centre.update(self.h_changes)
        centre.update(self.charge_changes)
        centre.update(self.stereo_changes)
        for pair in self.bond_stereo_changes:
            centre.update(m for m in pair if m in self.product_maps)
        return centre

    @property
    def changed_bonds(self) -> list[tuple[int, int, int, int]]:
        return [(c.a, c.b, c.product, c.reactant) for c in self.bond_changes]

    def summary(self) -> dict:
        """JSON-friendly description, used by the CLI and the analogue report."""
        return {
            "bonds_broken": [[c.a, c.b, c.product] for c in self.bond_changes if c.kind == "broken"],
            "bonds_formed": [[c.a, c.b, c.reactant] for c in self.bond_changes if c.kind == "formed"],
            "bond_order_changes": [[c.a, c.b, c.product, c.reactant]
                                   for c in self.bond_changes if c.kind == "order"],
            "h_changes": {str(m): list(v) for m, v in sorted(self.h_changes.items())},
            "charge_changes": {str(m): list(v) for m, v in sorted(self.charge_changes.items())},
            "stereo_changes": {str(m): list(v) for m, v in sorted(self.stereo_changes.items())},
            "bond_stereo_changes": {"-".join(map(str, sorted(k))): list(v)
                                    for k, v in self.bond_stereo_changes.items()},
            "reactant_only_atoms": sorted(self.reactant_only_maps),
            "reaction_centre": sorted(self.reaction_centre),
        }


def diff_mols(product: Chem.Mol, reactants: Chem.Mol) -> ReactionDiff:
    """Diff two fully-mapped molecules (retro sense). Maps must already be consistent.

    Every atom on both sides needs a unique non-zero map: bonds are keyed by
    map pairs, so two unmapped atoms would silently collapse into one.
    """
    for side, mol in (("product", product), ("reactants", reactants)):
        maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
        if 0 in maps or len(set(maps)) != len(maps):
            raise ValueError(f"{side}: every atom needs a unique non-zero map to be diffed")
    p_be = be_matrix(product)
    r_be = be_matrix(reactants, reference=p_be.bonds)
    p_maps = set(p_be.atoms)
    r_only = set(r_be.atoms) - p_maps

    bond_changes = []
    for pair in sorted(set(p_be.bonds) | set(r_be.bonds), key=lambda p: tuple(sorted(p))):
        po, ro = p_be.bonds.get(pair, 0), r_be.bonds.get(pair, 0)
        if po != ro:
            a, b = sorted(pair)
            bond_changes.append(BondChange(a, b, po, ro))

    h_changes, charge_changes, stereo_changes = {}, {}, {}
    for m in sorted(p_maps & set(r_be.atoms)):
        pa, ra = p_be.atoms[m], r_be.atoms[m]
        if pa.hs != ra.hs:
            h_changes[m] = (pa.hs, ra.hs)
        if pa.charge != ra.charge:
            charge_changes[m] = (pa.charge, ra.charge)
        if pa.stereo != ra.stereo:
            stereo_changes[m] = (pa.stereo, ra.stereo)
    bond_stereo_changes = {}
    for pair in set(p_be.bond_stereo) | set(r_be.bond_stereo):
        if pair <= p_maps and p_be.bond_stereo.get(pair) != r_be.bond_stereo.get(pair):
            bond_stereo_changes[pair] = (p_be.bond_stereo.get(pair), r_be.bond_stereo.get(pair))

    return ReactionDiff(
        product=p_be,
        reactants=r_be,
        product_maps=p_maps,
        reactant_only_maps=r_only,
        bond_changes=bond_changes,
        h_changes=h_changes,
        charge_changes=charge_changes,
        stereo_changes=stereo_changes,
        bond_stereo_changes=bond_stereo_changes,
    )


def diff_reaction(mapped_product: str, mapped_reactants: str) -> ReactionDiff:
    """:func:`diff_mols` on SMILES. Retro sense: ``product>>reactants``."""
    return diff_mols(parse(mapped_product), parse(mapped_reactants))


def mapping_problems(product: Chem.Mol, reactants: Chem.Mol) -> list[str]:
    """Why a mapped reaction cannot be diffed, or [] if it can.

    A product atom must be mapped, carry a unique map, and appear exactly once
    among the reactants with the same element. Reactant maps must be unique.
    Anything else would make the BE diff describe a reaction that does not
    conserve atoms, and op generation would then silently invent chemistry.
    """
    problems: list[str] = []
    p_maps = [a.GetAtomMapNum() for a in product.GetAtoms()]
    if any(m == 0 for m in p_maps):
        problems.append(f"{sum(m == 0 for m in p_maps)} product atom(s) unmapped")
    dup = sorted({m for m in p_maps if m and p_maps.count(m) > 1})
    if dup:
        problems.append(f"duplicate product maps {dup}")
    r_maps = [a.GetAtomMapNum() for a in reactants.GetAtoms() if a.GetAtomMapNum()]
    rdup = sorted({m for m in r_maps if r_maps.count(m) > 1})
    if rdup:
        problems.append(f"duplicate reactant maps {rdup}")
    r_atoms = atom_by_map(reactants)
    missing = sorted(m for m in p_maps if m and m not in r_atoms)
    if missing:
        problems.append(f"product maps absent from reactants {missing}")
    for pa in product.GetAtoms():
        ra = r_atoms.get(pa.GetAtomMapNum())
        if ra is not None and ra.GetAtomicNum() != pa.GetAtomicNum():
            problems.append(
                f"element mismatch at map {pa.GetAtomMapNum()}: "
                f"product {pa.GetSymbol()} vs reactant {ra.GetSymbol()}")
    return problems
