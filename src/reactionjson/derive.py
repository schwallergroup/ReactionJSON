"""Translate a mapped reaction's BE-matrix diff into a v2 atomic-op sequence.

Input: one mapped step in the retro sense, ``product>>reactants``. Output: a
list of v2 ops that ``execute_ops(ops, product, vocabulary="v2", kekulize=True)``
turns into those reactants.

Why the ops are derived *against the live molecule* rather than all at once
from a single static diff: the executor re-kekulizes before every op, and an
aromatic ring that the previous op left intact can come back with its double
bonds in different places. A plan computed once against the starting Kekulé
structure would then ladder the wrong ring bonds. So each op is chosen from the
diff between the *current* state (as the executor will see it) and the target,
applied with the real op functions, and the diff recomputed. The final plan is
still re-validated from scratch with :func:`execute_ops` (see ``roundtrip``),
which catches any drift between this mirror and the executor.

Translation rules (see README for the long form):

1. Bond orders go down first — ``break_bond`` once per order, highest component
   first, so a C=O → C–O is one op and severing a double bond is two. Breaking
   first frees the valence that later additions need.
   Then formal charges — ``set_formal_charge`` on each product atom whose charge
   differs. At this point every bond is at or below its target order, so the
   new charge's valence always fits; the H it refills are consumed by the bonds
   that follow.
2. Reactant-only atoms (maps absent from the product: leaving groups, the other
   half of a coupling, water, …) come in as ``add_group`` fragments, one per
   connected component, written in Kekulé form with every atom carrying its
   global map number so later ops can address it. The fragment is attached
   through one bond; any further bond between the component and the product
   atoms (a ring closure) is an ``add_bond`` afterwards. A component with no bond
   to any product atom (a whole co-reactant that never touches the product
   skeleton) is attached to a spare H-bearing atom and immediately severed.
3. Bond orders go up — a new bond is one ``add_bond`` with its full ``order``,
   an existing bond is raised one ``add_bond`` per order (the ladder).
4. Hydrogen counts that valence refilling did not already get right are set with
   ``set_explicit_h``.
5. Stereo last, once the constitution matches: ``set_stereocenter`` (R/S, or
   null to clear) for tetrahedral centres, tagged or not (a tag cleared by a
   break_bond, or never drawn on the product), ``set_bond_stereo`` for alkenes (a stale E/Z is cleared by stepping
   the bond down and up again).

What v2 cannot express is flagged, never silently dropped: a tetrahedral centre
the reactants need but that has no R/S configuration (relative-only stereo, such
as cis-1,4-disubstituted cyclohexane) is reported in ``flags`` as
``stereo_lost:<map>``, and the round trip then fails.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from reactionjson._be_matrix import ReactionDiff, diff_mols, mapping_problems
from reactionjson._chem import (
    atom_by_map,
    canonical_mapped,
    fragment_multiset,
    kekule_orders,
    kekulize_like,
    kekulized,
    parse,
    stereo_labels,
)
from reactionjson.executor import _resolve_map
from reactionjson.ops import (
    add_bond,
    add_group,
    break_bond,
    set_bond_stereo,
    set_explicit_h,
    set_formal_charge,
    set_stereocenter,
)


class OpDerivationError(ValueError):
    """The diff could not be expressed as v2 ops; the message says where."""


@dataclass
class OpsDerivation:
    """Result of :func:`derive_ops`."""

    ops: list[dict[str, Any]]
    product: str                  # canonical mapped product the ops apply to
    target: str                   # mapped reactants the ops must produce (fresh maps added)
    reactants: str                # mapped reactants as given
    diff: Optional[ReactionDiff]
    fresh_maps: dict[int, int] = field(default_factory=dict)  # reactant atom idx -> map given to an unmapped atom
    flags: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Op application — a mirror of execute_ops' per-op dispatch, restricted to v2.
# ─────────────────────────────────────────────────────────────────────────────

def apply_op(mol: Chem.Mol, op: dict[str, Any]) -> Chem.Mol:
    """Apply one v2 op dict to ``mol`` exactly as ``execute_ops`` would."""
    name = op["op"]
    r = lambda key: _resolve_map(mol, op[key])  # noqa: E731
    if name == "break_bond":
        return break_bond(mol, r("map_a"), r("map_b"))
    if name == "add_bond":
        return add_bond(mol, r("map_a"), r("map_b"), op.get("order", 1))
    if name == "add_group":
        return add_group(mol, r("map_idx"), op["fragment_smiles"], op.get("order", 1))
    if name == "set_explicit_h":
        return set_explicit_h(mol, r("map_idx"), op["n"])
    if name == "set_formal_charge":
        return set_formal_charge(mol, r("map_idx"), op["charge"])
    if name == "set_stereocenter":
        return set_stereocenter(mol, r("map_idx"), op["stereo"])
    if name == "set_bond_stereo":
        ref_a = r("map_ref_a") if op.get("map_ref_a") else None
        ref_b = r("map_ref_b") if op.get("map_ref_b") else None
        return set_bond_stereo(mol, r("map_a"), r("map_b"), op["stereo"], ref_a, ref_b)
    raise ValueError(f"not a v2 op: {name!r}")


def _executor_view(mol: Chem.Mol) -> Chem.Mol:
    """What the executor hands the next op: a Kekulé copy (best effort, like it)."""
    try:
        return kekulized(mol)
    except (Chem.KekulizeException, Chem.AtomKekulizeException, ValueError):
        return mol


# ─────────────────────────────────────────────────────────────────────────────
# Target preparation: fresh maps for unmapped reactant atoms.
# ─────────────────────────────────────────────────────────────────────────────

def _assign_fresh(reactants: Chem.Mol, start: int) -> tuple[Chem.Mol, dict[int, int]]:
    """Give every unmapped reactant atom a unique map >= ``start``."""
    rw = Chem.RWMol(reactants)
    used = {a.GetAtomMapNum() for a in rw.GetAtoms()}
    nxt, given = start, {}
    for a in rw.GetAtoms():
        if a.GetAtomMapNum() == 0:
            while nxt in used:
                nxt += 1
            a.SetAtomMapNum(nxt)
            used.add(nxt)
            given[a.GetIdx()] = nxt
    return rw.GetMol(), given


# ─────────────────────────────────────────────────────────────────────────────
# Fragment writing.
# ─────────────────────────────────────────────────────────────────────────────

def _fragment_smiles(target_kek: Chem.Mol, atoms: set[int], anchor: Optional[int],
                     order: int, *, spare_h: int = 0) -> str:
    """SMILES for ``add_group``: target atoms ``atoms`` plus ``*`` bonded to ``anchor``.

    Written from the Kekulé target so a component that is only part of an
    aromatic ring still parses. Every atom keeps its map (later ops reference
    them) and an explicit H count equal to the target's, so an atom that still
    awaits a ring-closing bond is written under-valent and ``add_bond`` later
    sees the spare valence instead of stripping an H.

    ``atoms`` may span several components: co-reactants that share no bond with
    the product (a counter-ion, Cl2, SOCl2) ride along dot-separated, which
    add_group accepts as long as there is exactly one ``*``. ``anchor=None``
    attaches through an explicit ``[H]`` instead, for a step that has nothing
    else to attach; ``spare_h`` lowers the anchor's count for the
    attach-then-sever trick (the bond to ``*`` occupies the slot of the H that
    break_bond will refill).
    """
    by_map = atom_by_map(target_kek)
    rw = Chem.RWMol()
    new_idx: dict[int, int] = {}
    for m in sorted(atoms):
        src = by_map[m]
        at = Chem.Atom(src.GetAtomicNum())
        at.SetFormalCharge(src.GetFormalCharge())
        at.SetIsotope(src.GetIsotope())
        at.SetAtomMapNum(m)
        at.SetNoImplicit(True)
        h = src.GetTotalNumHs() - (spare_h if m == anchor else 0)
        at.SetNumExplicitHs(max(h, 0))
        at.SetNumRadicalElectrons(0)
        new_idx[m] = rw.AddAtom(at)
    dummy = rw.AddAtom(Chem.Atom(0))
    order_type = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for b in target_kek.GetBonds():
        a1, a2 = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        if a1 in atoms and a2 in atoms:
            rw.AddBond(new_idx[a1], new_idx[a2], b.GetBondType())
    if anchor is None:
        h_idx = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(h_idx, dummy, Chem.BondType.SINGLE)
    else:
        rw.AddBond(new_idx[anchor], dummy, order_type[order])
    frag = rw.GetMol()
    frag.UpdatePropertyCache(strict=False)
    _copy_chirality(target_kek, frag, atoms)
    smi = Chem.MolToSmiles(frag, kekuleSmiles=True, canonical=True)
    if Chem.MolFromSmiles(smi) is None:
        raise OpDerivationError(f"fragment for atoms {sorted(atoms)} does not parse: {smi!r}")
    return smi


def _copy_chirality(src: Chem.Mol, dst: Chem.Mol, component: set[int]) -> None:
    """Carry tetrahedral tags from ``src`` onto the rebuilt fragment ``dst``.

    A tag is a parity relative to neighbour order, and the fragment lists
    neighbours in a different order (and has a ``*`` or nothing where product
    atoms were). We copy the tag and fix its parity by the permutation between
    the two neighbour orders; a neighbour missing from the fragment (a pending
    ring closure) makes the parity undefined here, and the stereo phase of the
    derivation then inverts it if needed — what matters is that a tag exists.
    """
    s_by = atom_by_map(src)
    d_by = atom_by_map(dst)
    for m in component:
        s_atom = s_by[m]
        tag = s_atom.GetChiralTag()
        if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            continue
        d_atom = d_by[m]
        s_nbrs = [b.GetOtherAtom(s_atom).GetAtomMapNum() for b in s_atom.GetBonds()]
        d_nbrs = []
        for b in d_atom.GetBonds():
            other = b.GetOtherAtom(d_atom)
            d_nbrs.append(other.GetAtomMapNum() if other.GetAtomicNum() else None)
        # The dummy stands in for whichever source neighbour is not in the component.
        missing = [n for n in s_nbrs if n not in component]
        d_nbrs = [(missing[0] if (n is None and missing) else n) for n in d_nbrs]
        common = [n for n in s_nbrs if n in d_nbrs]
        if len(common) == len(s_nbrs) == len(d_nbrs):
            perm = [s_nbrs.index(n) for n in d_nbrs]
            parity = _perm_parity(perm)
            flip = {Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW}
            d_atom.SetChiralTag(tag if parity == 0 else flip[tag])
        else:
            d_atom.SetChiralTag(tag)


def _perm_parity(perm: list[int]) -> int:
    perm = list(perm)
    parity = 0
    for i in range(len(perm)):
        while perm[i] != i:
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
            parity ^= 1
    return parity


def _attachment_bonds(t_kek: Chem.Mol, comp: set[int],
                      present: set[int]) -> list[tuple[int, int, int]]:
    """(component atom, present atom, Kekulé order) for every bond out of ``comp``.

    Sorted so single bonds come first: attaching through a single bond keeps the
    fragment anchor's H count untouched (add_group only adjusts it for order > 1).
    """
    out = []
    for b in t_kek.GetBonds():
        a1, a2 = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        order = int(round(b.GetBondTypeAsDouble()))
        if a1 in comp and a2 in present:
            out.append((a1, a2, order))
        elif a2 in comp and a1 in present:
            out.append((a2, a1, order))
    return sorted(out, key=lambda t: (t[2], t[1], t[0]))


def _components(target: Chem.Mol, maps: set[int]) -> list[set[int]]:
    """Connected components of the target restricted to ``maps``."""
    by_map = atom_by_map(target)
    left, comps = set(maps), []
    while left:
        seed = min(left)
        comp, stack = set(), [seed]
        while stack:
            m = stack.pop()
            if m in comp:
                continue
            comp.add(m)
            for nb in by_map[m].GetNeighbors():
                nm = nb.GetAtomMapNum()
                if nm in left and nm not in comp:
                    stack.append(nm)
        comps.append(comp)
        left -= comp
    return comps


# ─────────────────────────────────────────────────────────────────────────────
# The planner.
# ─────────────────────────────────────────────────────────────────────────────

def _reread(mol: Chem.Mol) -> Chem.Mol:
    """``mol`` as the next reader of the executor's output string will see it.

    An op that closes a ring (add_bond, or add_group + add_bond) sanitizes a
    molecule whose ring info predates the new bond, so a freshly closed aromatic
    ring can stay in Kekulé form on the op's return value while the SMILES it
    writes re-parses as aromatic. Comparisons against the target must use the
    re-parsed form; the op sequence itself keeps working on the live molecule.
    """
    return parse(Chem.MolToSmiles(mol))


def _constitution(mol: Chem.Mol) -> list[str]:
    """Mapped, stereo-free fragment keys: equal iff same atoms, bonds, H, charge."""
    return sorted(Chem.MolToSmiles(f, isomericSmiles=False)
                  for f in Chem.GetMolFrags(_reread(mol), asMols=True))


def derive_ops(
    mapped_product: str,
    mapped_reactants: str,
    *,
    fresh_start: Optional[int] = None,
    max_ops: int = 400,
) -> OpsDerivation:
    """Derive v2 ops taking ``mapped_product`` to ``mapped_reactants`` (retro).

    ``mapped_reactants`` may contain unmapped atoms (they get fresh maps from
    ``fresh_start``, default one above every map in the reaction). The returned
    ``target`` is the reactant set the ops produce, which differs from the input
    only by those fresh maps.
    """
    product_smi = canonical_mapped(mapped_product)
    product = parse(product_smi)
    reactants = parse(mapped_reactants)
    result = OpsDerivation(ops=[], product=product_smi, target=mapped_reactants,
                           reactants=mapped_reactants, diff=None)

    all_maps = [a.GetAtomMapNum() for a in product.GetAtoms()] + \
               [a.GetAtomMapNum() for a in reactants.GetAtoms()]
    start = fresh_start or (max(all_maps + [0]) + 1)
    target, result.fresh_maps = _assign_fresh(reactants, start)

    problems = mapping_problems(product, target)
    if problems:
        result.error = "invalid mapping: " + "; ".join(problems)
        return result

    result.target = Chem.MolToSmiles(target)
    target = parse(result.target)
    result.diff = diff_mols(product, target)

    p_maps = set(atom_by_map(product))
    comps = _components(target, set(atom_by_map(target)) - p_maps)
    # Components that share a bond with the product are attached through it;
    # the rest (co-reactants that never touch the product skeleton) are
    # passengers on the first add_group fragment.
    pending = [c for c in comps if _attachment_bonds(target, c, p_maps)]
    passengers: set[int] = set().union(*[c for c in comps if not _attachment_bonds(target, c, p_maps)])
    target_constitution = _constitution(target)
    t_labels, t_bond_labels = stereo_labels(target)
    t_by_map = atom_by_map(target)

    current = product            # sanitized molecule after the last op
    ops: list[dict[str, Any]] = []
    tried_stereo: set = set()

    def emit(op: dict[str, Any]) -> None:
        nonlocal current
        view = _executor_view(current) if ops else kekulized(current)
        try:
            current = apply_op(view, op)
        except (ValueError, KeyError, RuntimeError, Chem.AtomValenceException,
                Chem.KekulizeException, Chem.AtomKekulizeException) as exc:
            raise OpDerivationError(f"op {len(ops)} {op} failed: {type(exc).__name__}: {exc}") from exc
        if any(a.GetAtomicNum() == 1 for a in current.GetAtoms()):
            # An explicit [H] anchor (see _fragment_smiles) stays a graph atom in
            # the executor; any re-parse folds it into the H count, so do that here.
            current = Chem.RemoveHs(current)
        ops.append(op)

    try:
        while len(ops) < max_ops:
            view = _executor_view(current) if ops else kekulized(current)
            present = set(atom_by_map(view))
            s_orders = kekule_orders(view)
            t_kek = kekulize_like(target, s_orders)
            t_orders = {p: o for p, o in kekule_orders(t_kek).items() if p <= present}

            # 1. Lower bond orders (break_bond, one order per op).
            lower = sorted((tuple(sorted(p)) for p, o in s_orders.items()
                            if t_orders.get(p, 0) < o))
            if lower:
                a, b = lower[0]
                bond = view.GetBondBetweenAtoms(_resolve_map(view, a), _resolve_map(view, b))
                if bond.GetBondType() == Chem.BondType.DATIVE:
                    # break_bond hands any non-single bond to change_bond_order,
                    # which cannot take a dative bond to zero; v2 has no other op.
                    raise OpDerivationError(
                        f"the mapping severs the dative bond {a}->{b}, which no v2 op can break")
                emit({"op": "break_bond", "map_a": a, "map_b": b})
                continue

            # 1b. Formal charges, with every bond now at or below its target order.
            cur_by_map = atom_by_map(current)
            charge_fix = sorted(m for m, a in cur_by_map.items() if m in t_by_map
                                and a.GetFormalCharge() != t_by_map[m].GetFormalCharge())
            if charge_fix:
                m = charge_fix[0]
                emit({"op": "set_formal_charge", "map_idx": m,
                      "charge": t_by_map[m].GetFormalCharge()})
                continue

            # 2. Bring in reactant-only components.
            if pending:
                comp = pending.pop(0)
                c_atom, p_atom, order = _attachment_bonds(t_kek, comp, present)[0]
                frag = _fragment_smiles(t_kek, comp | passengers, c_atom, order)
                if passengers:
                    result.flags.append(f"passenger_coreactants:{min(passengers)}")
                passengers = set()
                op: dict[str, Any] = {"op": "add_group", "map_idx": p_atom, "fragment_smiles": frag}
                if order != 1:
                    op["order"] = order
                emit(op)
                continue
            if passengers:
                _introduce_passengers(emit, view, t_kek, passengers, result)
                passengers = set()
                continue

            # 3. Raise bond orders (new bond: one add_bond of full order; existing: ladder).
            raise_ = sorted((tuple(sorted(p)), s_orders.get(p, 0), o) for p, o in t_orders.items()
                            if o > s_orders.get(p, 0))
            if raise_:
                (a, b), have, want = raise_[0]
                op = {"op": "add_bond", "map_a": a, "map_b": b}
                if have == 0 and want > 1:
                    op["order"] = want
                emit(op)
                continue

            # 4. Hydrogen counts.
            cur_by_map = atom_by_map(current)
            h_fix = sorted(m for m, a in cur_by_map.items()
                           if a.GetTotalNumHs() != t_by_map[m].GetTotalNumHs())
            if h_fix:
                m = h_fix[0]
                emit({"op": "set_explicit_h", "map_idx": m, "n": t_by_map[m].GetTotalNumHs()})
                continue

            if _constitution(current) != target_constitution:
                raise OpDerivationError(
                    "constitution still differs after bond/H phases: "
                    f"have {_constitution(current)} want {target_constitution}")

            # 5. Stereo.
            if _stereo_step(emit, current, t_labels, t_bond_labels, target, tried_stereo, result):
                continue
            break
        else:
            raise OpDerivationError(f"exceeded {max_ops} ops without reaching the target")
    except OpDerivationError as exc:
        result.ops = ops
        result.error = str(exc)
        return result

    result.ops = ops
    return result


def _introduce_passengers(emit, view: Chem.Mol, t_kek: Chem.Mol, atoms: set[int],
                          result: OpsDerivation) -> None:
    """Introduce co-reactants in a step that attaches no other fragment.

    add_group needs an anchor. Preferably the co-reactant is grafted onto a
    spare H-bearing, stereo-free product atom through one of its own H-bearing
    atoms and the graft bond severed at once: break_bond refills an H on both
    ends, so each side ends with its original count. A co-reactant with no H at
    all (Na+, Cl2) instead rides on an explicit ``[H]`` grafted onto the host,
    which only moves one of the host's hydrogens into the graph.
    """
    t_by = atom_by_map(t_kek)
    host = next((a.GetAtomMapNum() for a in view.GetAtoms()
                 if a.GetTotalNumHs() > 0 and a.GetFormalCharge() == 0
                 and a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                 and not any(b.GetBondType() == Chem.BondType.DOUBLE
                             and b.GetStereo() != Chem.BondStereo.STEREONONE
                             for b in a.GetBonds())), None)
    if host is None:
        raise OpDerivationError(
            f"co-reactant atoms {sorted(atoms)} share no bond with the product and "
            "the product has no stereo-free H-bearing atom to graft them through")
    anchor = next((m for m in sorted(atoms)
                   if t_by[m].GetTotalNumHs() > 0 and t_by[m].GetFormalCharge() == 0
                   and t_by[m].GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED), None)
    if anchor is not None:
        frag = _fragment_smiles(t_kek, atoms, anchor, 1, spare_h=1)
        emit({"op": "add_group", "map_idx": host, "fragment_smiles": frag})
        emit({"op": "break_bond", "map_a": host, "map_b": anchor})
    else:
        frag = _fragment_smiles(t_kek, atoms, None, 1)
        emit({"op": "add_group", "map_idx": host, "fragment_smiles": frag})
    result.flags.append(f"passenger_coreactants:{min(atoms)}")


def _stereo_step(emit, current: Chem.Mol, t_labels: dict, t_bond_labels: dict,
                 target: Chem.Mol, tried: set, result: OpsDerivation) -> bool:
    """Emit one stereo op if any descriptor differs; return whether one was emitted."""
    c_labels, c_bond_labels = stereo_labels(_reread(current))
    t_by = atom_by_map(target)
    c_by = atom_by_map(current)
    for m in sorted(set(c_labels) | set(t_labels)):
        want, have = t_labels.get(m), c_labels.get(m)
        if want == have:
            continue
        key = ("atom", m, want, have)
        if key in tried:
            continue
        tried.add(key)
        if have and not want:
            if t_by[m].GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                emit({"op": "set_stereocenter", "map_idx": m, "stereo": None})
                return True
            continue
        if want:
            # Real CIP (not the map-resolved label), since that is what the op reads.
            cip = _real_cip(target, m)
            if cip:
                emit({"op": "set_stereocenter", "map_idx": m, "stereo": cip})
                return True
            result.flags.append(f"stereo_lost:{m}")
            continue
    for pair in sorted(set(c_bond_labels) | set(t_bond_labels), key=lambda p: tuple(sorted(p))):
        want, have = t_bond_labels.get(pair), c_bond_labels.get(pair)
        if want == have:
            continue
        a, b = sorted(pair)
        key = ("bond", a, b, want, have)
        if key in tried:
            continue
        tried.add(key)
        if want:
            for letter in (want, "Z" if want == "E" else "E"):
                op = {"op": "set_bond_stereo", "map_a": a, "map_b": b, "stereo": letter}
                try:
                    trial = apply_op(_executor_view(current), op)
                except (ValueError, RuntimeError):
                    continue
                if stereo_labels(_reread(trial))[1].get(pair) == want:
                    emit(op)
                    return True
            result.flags.append(f"bond_stereo_unset:{a}-{b}")
            continue
        # A stale E/Z with no op to clear it: step the bond down and back up,
        # which change_bond_order does with its stereo cleared.
        emit({"op": "break_bond", "map_a": a, "map_b": b})
        emit({"op": "add_bond", "map_a": a, "map_b": b})
        return True
    return False


def _real_cip(mol: Chem.Mol, map_num: int) -> Optional[str]:
    """Chemical CIP label ('R'/'S') of the atom with ``map_num``, or None."""
    m = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(m)
    atom = atom_by_map(m)[map_num]
    code = atom.GetProp("_CIPCode").upper() if atom.HasProp("_CIPCode") else None
    return code if code in ("R", "S") else None


def derive_ops_for_reaction(mapped_retro_reaction: str, **kwargs) -> OpsDerivation:
    """:func:`derive_ops` on a ``product>>reactants`` string."""
    product, reactants = mapped_retro_reaction.split(">>", 1)
    return derive_ops(product, reactants, **kwargs)


def ops_copy(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return deepcopy(ops)


__all__ = [
    "OpsDerivation",
    "OpDerivationError",
    "apply_op",
    "derive_ops",
    "derive_ops_for_reaction",
    "fragment_multiset",
]
