"""
LTN logic for the Faster R-CNN detector (thesis Chapter 6). Version 5 (call of 6 Nov 2026).
Pure PyTorch: no model, no data loading. Notation as in SymMedNet (paper_bmvc_2.pdf, sections 3.3-3.4, appendix B.9).

Individuals x = the regions of interest (RoIs) sampled by the box head (512 per image).

Groundings in the detector (the class head is ONE softmax over 5 classes: 0 background, 1 mass benign,
2 mass malignant, 3 calc benign, 4 calc malignant):
    isOfClass(x, l_Benign)        = p1(x) + p3(x)
    isOfClass(x, l_Malignant)     = p2(x) + p4(x)
    has_mass(x, c_mass)           = p1(x) + p2(x)
    has_calc(x, c_calc)           = p3(x) + p4(x)
    has_shape(x, c_a)             = softmax of the shape head, component a          (c_shape_ov, c_shape_rd, c_shape_ir)
    has_margin(x, c_a)            = sigmoid of the margin head, component a          (c_margin_circ, _obs, _ml, _ind, _spic)
    is_high_density(x, c_high_dens) = softmax of the density head, component "high"
    has_morphology(x, c_a)        = sigmoid of the morphology head, component a      (c_morph_t_ben, _ch, _fp, _amo, _fl)
    has_distribution(x, c_a)      = softmax of the distribution head, component a    (c_distr_clu, _seg, _lin, _reg, _dif)

Connectives (product logic): not a = 1 - a, a and b = ab, a or b = a + b - ab, a => b = 1 - a + ab.
Quantifier: forall = ApME_p (p = 2 fixed in our runs, or the SymMedNet schedule 2 -> 10).

Knowledge base of one batch, K = K_class U K_facts U K_axioms; every loss is 1 - Sat:
    phi_class      forall Diag(x, l_k). isOfClass(x, l_k)                       loss_class_sat
    phi1, phi2     forall Diag(x, l_mass). has_mass(x, c_mass)  (and calc)      loss_find   (only with concept facts)
    phi3-phi21     forall Diag(x, l_a). has_attr_h(x, l_a)                      loss_attr   (the "+C" runs)
    psi3-psi16     forall Diag(x, l_k): body => isOfClass(x, l_k)               loss_clin   (the "Ax" runs)
psi1, psi2, psi15 conclude Normal: no Normal class in the detector, so they are refused (call of 6 Nov: start with
benign and malignant).
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = 5
EPS = 1e-4          # keeps 1 - truth away from 0, so ApME has a finite gradient when a formula is fully satisfied

BG, MB, MM, CB, CM = 0, 1, 2, 3, 4
CLASS_SETS = {"l_Benign": (MB, CB), "l_Malignant": (MM, CM)}           # isOfClass(x, l_k) and Diag(x, l_k)
FINDING_SETS = {"c_mass": (MB, MM), "c_calc": (CB, CM)}                 # has_mass(x, c_mass), has_calc(x, c_calc)
FAMILY_SETS = {"MASS": (MB, MM), "CALC": (CB, CM)}


# --------------------------------------------------------------------------------------------------
# Connectives (product logic)
# --------------------------------------------------------------------------------------------------
# negation: not a = 1 - a
def Not(a: torch.Tensor) -> torch.Tensor:
    return 1.0 - a


# product t-norm: a and b = a * b (n-ary)
def And(*xs: torch.Tensor) -> torch.Tensor:
    out = xs[0]
    for x in xs[1:]:
        out = out * x
    return out


# probabilistic sum (dual of the product t-norm), used for 'or' of independent atoms
def Or(*xs: torch.Tensor) -> torch.Tensor:
    """Probabilistic sum: 1 - prod(1 - x)."""
    out = 1.0 - xs[0]
    for x in xs[1:]:
        out = out * (1.0 - x)
    return 1.0 - out


def Implies(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reichenbach implication 1 - a + ab."""
    # Reichenbach implication of product logic (as SymMedNet / LTNtorch). Note dI/da = b - 1 <= 0 and dI/db = a:
    # besides raising the consequent b, a rule can also be satisfied by lowering its antecedent a.
    return 1.0 - a + a * b


# --------------------------------------------------------------------------------------------------
# Quantifier and aggregation
# --------------------------------------------------------------------------------------------------
def apme(truths: torch.Tensor, p: float, mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """forall x: ApME_p = 1 - ( mean_i (1 - t_i)^p )^(1/p). `mask` = the guard Diag (the domain).
    Returns None on an empty domain: the formula is vacuous in this batch and is skipped, not counted as satisfied."""
    t = truths.reshape(-1)
    if mask is not None:
        t = t[mask.reshape(-1)]
    if t.numel() == 0:
        return None
    # truths scaled to [0, 1 - EPS]: at t = 1 the gradient of (mean (1 - t)^p)^(1/p) is undefined (0 ** (1/p - 1))
    t = (1.0 - EPS) * t
    # 1 - power mean of the errors (1 - t): p = 1 is the mean of the truths, p -> inf tends to the minimum (strict forall)
    return 1.0 - torch.mean((1.0 - t) ** p) ** (1.0 / p)


def sat_agg(sats: Sequence[Optional[torch.Tensor]], p: float) -> Optional[torch.Tensor]:
    """SatAgg of a set of formulas = ApME over their truth values (vacuous formulas left out)."""
    vals = [s for s in sats if s is not None]
    if not vals:
        return None
    return apme(torch.stack(vals), p)


def p_schedule(epoch: int, epochs: int, values=(2, 4, 6, 8, 10),
               starts=(0.0, 1 / 3, 0.6, 0.8, 14 / 15)) -> float:
    """SymMedNet's schedule of p over 150 epochs (2, 4, 6, 8, 10), at the same fractions of training."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    frac = epoch / epochs
    p = values[0]
    # the last stage whose start fraction has been reached gives p
    for v, s in zip(values, starts):
        if frac >= s:
            p = v
    return float(p)


# --------------------------------------------------------------------------------------------------
# Groundings
# --------------------------------------------------------------------------------------------------
PREDICATE_HEAD = {"has_shape": "mass_shape", "has_margin": "mass_margin", "is_high_density": "mass_density",
                  "has_morphology": "calc_morph", "has_distribution": "calc_distr"}
CONSTANTS = {   # constant c_a -> (head, value of the head)
    "c_shape_ov": ("mass_shape", "oval"), "c_shape_rd": ("mass_shape", "round"), "c_shape_ir": ("mass_shape", "irregular"),
    "c_margin_circ": ("mass_margin", "circumscribed"), "c_margin_obs": ("mass_margin", "obscured"),
    "c_margin_ml": ("mass_margin", "microlobulated"), "c_margin_ind": ("mass_margin", "indistinct"),
    "c_margin_spic": ("mass_margin", "spiculated"),
    "c_high_dens": ("mass_density", "high"),
    "c_morph_t_ben": ("calc_morph", "typically_benign"), "c_morph_ch": ("calc_morph", "coarse_heterogeneous"),
    "c_morph_fp": ("calc_morph", "pleomorphic"), "c_morph_amo": ("calc_morph", "amorphous"),
    "c_morph_fl": ("calc_morph", "fine_linear_branching"),
    "c_distr_clu": ("calc_distr", "clustered"), "c_distr_seg": ("calc_distr", "segmental"),
    "c_distr_lin": ("calc_distr", "linear"), "c_distr_reg": ("calc_distr", "regional"),
    "c_distr_dif": ("calc_distr", "diffuse"),
}


def class_predicates(class_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """isOfClass(x, l_k), has_mass(x, c_mass), has_calc(x, c_calc) from the softmax of the class head."""
    p = F.softmax(class_logits, dim=-1)
    # the 4 lesion classes are {mass, calc} x {benign, malignant}, so every class predicate of the paper is the sum
    # of two softmax outputs (exact, the outputs are exclusive); background belongs to none of them
    return {"probs": p,
            "l_Benign": p[:, MB] + p[:, CB], "l_Malignant": p[:, MM] + p[:, CM],
            "c_mass": p[:, MB] + p[:, MM], "c_calc": p[:, CB] + p[:, CM]}


def head_probs(attr_logits: Dict[str, torch.Tensor], activation: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Truth values of every concept head: softmax for single-valued concepts, sigmoid for multi-label ones."""
    out = {}
    for h, z in attr_logits.items():
        act = activation[h]
        if act == "softmax":
            out[h] = F.softmax(z, dim=-1)
        elif act == "sigmoid":
            out[h] = torch.sigmoid(z)
        else:
            raise ValueError(f"head {h}: activation {act!r}")
    return out


def has_value(attr_probs: torch.Tensor, value_idx: Union[int, Sequence[int]], activation: str = "softmax"
              ) -> torch.Tensor:
    """has_attr_h(x, c_a); a list of constants is a disjunction: for a softmax head the exact sum of the
    (exclusive) probabilities, for a sigmoid head the probabilistic sum."""
    if isinstance(value_idx, int):
        return attr_probs[:, value_idx]
    idx = list(value_idx)
    if activation == "softmax":
        # softmax outputs are exclusive: P(a or b) = P(a) + P(b) exactly
        return attr_probs[:, idx].sum(dim=-1).clamp(max=1.0)
    # sigmoid outputs are independent: probabilistic sum
    return Or(*[attr_probs[:, i] for i in idx])


def literal(pred: str, const: Union[str, Sequence[str]], cls: Dict[str, torch.Tensor],
            attr_probs: Dict[str, torch.Tensor], value_index, activation) -> torch.Tensor:
    """Truth of the atom pred(x, const) on every region (const may be a list = disjunction, same head)."""
    # has_mass / has_calc come from the class head; every other predicate from its concept head
    if pred in ("has_mass", "has_calc"):
        return cls[const]
    head = PREDICATE_HEAD[pred]
    consts = [const] if isinstance(const, str) else list(const)
    # an axiom that uses a constant of another head fails here instead of reading the wrong output
    for c in consts:
        if CONSTANTS[c][0] != head:
            raise KeyError(f"constant {c} does not belong to predicate {pred} (head {head})")
    idx = [value_index[head][CONSTANTS[c][1]] for c in consts]
    return has_value(attr_probs[head], idx[0] if len(idx) == 1 else idx, activation[head])


# --------------------------------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------------------------------
def class_fact_loss(class_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy (class_loss="ce", the baseline): what torchvision computes; not added twice."""
    return F.cross_entropy(class_logits, labels)


CLASS_AGGS = ("all", "per_class", "balanced", "geo")


def class_facts_sat(class_logits: torch.Tensor, labels: torch.Tensor, p: float, agg: str = "geo",
                    bg_weight: float = 0.5) -> Optional[torch.Tensor]:
    """phi_class = forall Diag(x, l_k). isOfClass(x, l_k), truth t(x) = p_label(x); the loss is 1 - Sat.
    The options differ only in how the truths of the regions are aggregated into Sat:
       "all"        ApME over all regions in one clause (SymMedNet). In the detector almost all regions are background:
                    calcifications were lost (runs S0, 4 Nov).
       "per_class"  ApME per class, then SatAgg: background weighs 1/5, far too little (runs S0pc).
       "balanced"   ApME per class, then a weighted pMeanError: background bg_weight, each lesion class
                    (1 - bg_weight) / 4 (with 0.5: SymMedNet's batch balance, 1/2 Normal).
       "geo"        geometric mean of the truths, exp(mean log t) = the product over the regions (LTN's product /
                    log-product aggregator, as in Faster-LTN); 1 - Sat = 1 - exp(-CE)."""
    if agg == "geo":
        # log p_label(x) of every region; its mean over the regions is -CE, so Sat = exp(-CE): the same minimiser as
        # the baseline's cross-entropy, with the gradient scaled by Sat (checked by test_class_facts_geo_*)
        logt = F.log_softmax(class_logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
        return torch.exp(logt.mean())
    # truth of phi_class on every region = probability of its labelled class (background regions included)
    t = F.softmax(class_logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
    if agg == "all":
        return apme(t, p)
    k_all = class_logits.shape[-1]
    if agg == "per_class":
        # one ApME per class present in the batch, then SatAgg (ApME) over the classes
        return sat_agg([apme(t, p, mask=labels == k) for k in range(k_all)], p)
    if agg == "balanced":
        if not 0.0 < bg_weight < 1.0:
            raise ValueError(f"bg_weight {bg_weight}: must be in (0, 1)")
        # weighted power mean of the errors of the class clauses: background bg_weight, the 4 lesion classes the rest
        errs, ws = [], []
        for k in range(k_all):
            s_k = apme(t, p, mask=labels == k)
            if s_k is not None:
                errs.append((1.0 - s_k) ** p)
                ws.append(bg_weight if k == BG else (1.0 - bg_weight) / (k_all - 1))
        w = torch.tensor(ws, device=t.device, dtype=t.dtype)
        return 1.0 - ((torch.stack(errs) * w).sum() / w.sum()) ** (1.0 / p)
    raise ValueError(f"class_agg {agg!r}: use one of {CLASS_AGGS}")


def finding_facts_sat(cls: Dict[str, torch.Tensor], labels: torch.Tensor, p: float) -> Dict[str, Optional[torch.Tensor]]:
    """phi1 = forall Diag(x, l_mass). has_mass(x, c_mass); phi2 = forall Diag(x, l_calc). has_calc(x, c_calc).
    l_mass = the region is on a mass (label 1 or 2), l_calc = on a calcification (3 or 4)."""
    out = {}
    for name, const in (("phi1_has_mass", "c_mass"), ("phi2_has_calc", "c_calc")):
        ks = FINDING_SETS[const]
        # Diag(x, l_mass): regions whose label is mass benign or mass malignant (same for calc)
        out[name] = apme(cls[const], p, mask=(labels == ks[0]) | (labels == ks[1]))
    return out


def attribute_fact_truths(attr_probs: torch.Tensor, target: torch.Tensor, activation: str = "softmax"
                          ) -> torch.Tensor:
    """Truth of the concept facts of one head on every region.
       softmax: one fact per region, has_attr_h(x, l_a) for the annotated value(s) (several = disjunction). [N]
       sigmoid: one fact per region and value: has_attr_h(x, l_a) if a is annotated, not has_attr_h(x, l_a)
                otherwise (CBIS lists every value of the lesion). [N, K]"""
    if activation == "softmax":
        # softmax head: truth = probability mass on the annotated value(s)
        return (attr_probs * target).sum(dim=-1).clamp(max=1.0)
    # sigmoid head, value by value: p if the value is annotated, 1 - p if not (negative fact). About 4 of 5
    # values are negative, so they dominate one ApME over regions x values (see the v5 concept evaluation)
    return target * attr_probs + (1.0 - target) * (1.0 - attr_probs)


def head_facts_sat(probs: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, p: float, activation: str
                   ) -> Optional[torch.Tensor]:
    """forall Diag(x, l_a). has_attr_h(x, l_a) for one head (ApME over the annotated regions; for a sigmoid head over
    the annotated regions x values)."""
    t = attribute_fact_truths(probs, y, activation)
    # sigmoid head: truths are [N, K], the region mask is repeated over the K values
    mask = valid if t.dim() == 1 else valid[:, None].expand_as(t)
    return apme(t, p, mask=mask)


def attribute_facts_sat(attr_probs: Dict[str, torch.Tensor], targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                        p: float, activation: Dict[str, str]
                        ) -> Tuple[Optional[torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
    per_head = {h: head_facts_sat(pr, targets[h][0], targets[h][1], p, activation[h]) for h, pr in attr_probs.items()}
    return sat_agg(list(per_head.values()), p), per_head


def attribute_ce_loss(attr_logits: Dict[str, torch.Tensor], targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                      activation: Dict[str, str], n_total: Optional[int] = None
                      ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Run BC (R50+C, no logic): cross-entropy on the concept heads (BCE for sigmoid heads), same regions as the facts.
    n_total=N: sum over the annotated regions / N (all sampled regions), as torchvision's box loss."""
    per_head, log = {}, {}
    for head, z in attr_logits.items():
        y, valid = targets[head]
        if valid.any():
            if activation[head] == "softmax":
                logp = F.log_softmax(z[valid], dim=-1)
                # -log of the probability of the annotated value(s); several annotated values = their sum (disjunction)
                ce = -torch.logsumexp(logp.masked_fill(y[valid] <= 0, float("-inf")), dim=-1)
            else:
                ce = F.binary_cross_entropy_with_logits(z[valid], y[valid], reduction="none").mean(dim=-1)
            log[f"ce_attr/{head}"] = float(ce.detach().mean())
            per_head[head] = ce.mean() if n_total is None else ce.sum() / n_total
    if not per_head:
        return None, log
    if n_total is None:
        return torch.stack(list(per_head.values())).mean(), log
    return torch.stack(list(per_head.values())).sum() / len(attr_logits), log


# --------------------------------------------------------------------------------------------------
# Axioms psi (SymMedNet, appendix B.9.1), written as in the paper:
#     forall x Diag(x, guard): body_1 and ... and not neg_1 ... => isOfClass(x, head)
# body / nots: (predicate, constant) or (predicate, [constants]) = disjunction of constants of the same predicate.
# --------------------------------------------------------------------------------------------------
# 'not regional and not diffuse', shared by psi7, psi8 and psi13
_NOT_REG_DIF = [("has_distribution", "c_distr_reg"), ("has_distribution", "c_distr_dif")]
AXIOMS = {
    # Normal patterns: need a Normal class (not in the detector): refused by resolve_axioms
    "psi1":  dict(guard="l_Normal", body=[("has_mass", "c_mass")],
                  nots=[("has_shape", "c_shape_ir"), ("is_high_density", "c_high_dens")], head="l_Normal"),
    "psi2":  dict(guard="l_Normal", body=[("has_mass", "c_mass")], nots=[("is_high_density", "c_high_dens")],
                  head="l_Normal"),
    "psi15": dict(guard="l_Normal", body=[("has_calc", "c_calc"), ("has_distribution", "c_distr_dif")], nots=[],
                  head="l_Normal"),
    # Benign patterns
    "psi3":  dict(guard="l_Benign", body=[("has_mass", "c_mass"), ("has_margin", ["c_margin_obs", "c_margin_circ"])],
                  nots=[], head="l_Benign"),
    "psi14": dict(guard="l_Benign", body=[("has_calc", "c_calc"), ("has_morphology", "c_morph_fl"),
                                          ("has_distribution", "c_distr_clu")], nots=[], head="l_Benign"),
    "psi16": dict(guard="l_Benign", body=[("has_calc", "c_calc"), ("has_distribution", "c_distr_reg")], nots=[],
                  head="l_Benign"),
    # Malignant patterns
    "psi4":  dict(guard="l_Malignant", body=[("has_mass", "c_mass"), ("has_margin", "c_margin_ml")], nots=[],
                  head="l_Malignant"),
    "psi5":  dict(guard="l_Malignant", body=[("has_mass", "c_mass"), ("has_margin", "c_margin_ind")], nots=[],
                  head="l_Malignant"),
    "psi6":  dict(guard="l_Malignant", body=[("has_mass", "c_mass"), ("has_shape", "c_shape_ir")], nots=[],
                  head="l_Malignant"),
    "psi7":  dict(guard="l_Malignant", body=[("has_calc", "c_calc"), ("has_morphology", "c_morph_amo")],
                  nots=_NOT_REG_DIF, head="l_Malignant"),
    "psi8":  dict(guard="l_Malignant", body=[("has_calc", "c_calc"), ("has_morphology", "c_morph_fp")],
                  nots=_NOT_REG_DIF, head="l_Malignant"),
    "psi9":  dict(guard="l_Malignant", body=[("has_mass", "c_mass"), ("is_high_density", "c_high_dens")], nots=[],
                  head="l_Malignant"),
    "psi10": dict(guard="l_Malignant", body=[("has_mass", "c_mass"), ("has_margin", "c_margin_spic")], nots=[],
                  head="l_Malignant"),
    "psi11": dict(guard="l_Malignant", body=[("has_calc", "c_calc"), ("has_morphology", "c_morph_fl"),
                                             ("has_distribution", "c_distr_seg")], nots=[], head="l_Malignant"),
    "psi12": dict(guard="l_Malignant", body=[("has_calc", "c_calc"), ("has_morphology", "c_morph_fl"),
                                             ("has_distribution", "c_distr_lin")], nots=[], head="l_Malignant"),
    "psi13": dict(guard="l_Malignant", body=[("has_calc", "c_calc"), ("has_morphology", "c_morph_ch")],
                  nots=_NOT_REG_DIF, head="l_Malignant"),
    # our proposals (LTN_axioms_v4.xlsx, sheet "Proposed additions"): not in the default set
    "N1":    dict(guard="l_Benign", body=[("has_calc", "c_calc"), ("has_morphology", "c_morph_t_ben")],
                  nots=[("has_morphology", c) for c in ("c_morph_amo", "c_morph_fp", "c_morph_fl", "c_morph_ch")],
                  head="l_Benign"),
    "N2":    dict(guard="l_Malignant", body=[("has_calc", "c_calc"), ("has_morphology", "c_morph_fl")], nots=[],
                  head="l_Malignant"),
    "N3":    dict(guard="l_Benign", body=[("has_calc", "c_calc"), ("has_distribution", "c_distr_dif")], nots=[],
                  head="l_Benign"),
}
# psi1, psi2, psi15: refused by resolve_axioms
NORMAL_AXIOMS = tuple(n for n, a in AXIOMS.items() if a["head"] == "l_Normal")
_KB14 = tuple(f"psi{i}" for i in range(3, 15))
AXIOM_SETS = {
    "kb16": _KB14 + ("psi16",),          # call of 6 Nov: the 16 axioms of SymMedNet without the Normal ones (13)
    "kb14": _KB14,                       # SymMedNet's default budget without the Normal ones (12)
    "kb11": tuple(f"psi{i}" for i in range(3, 12)),
    "kb16_N1": _KB14 + ("psi16", "N1"),
}
DEFAULT_AXIOMS = AXIOM_SETS["kb16"]


def axiom_text(name: str) -> str:
    """The axiom in the paper's notation, e.g.
    psi6: forall x Diag(x, l_Malignant): has_mass(x, c_mass) and has_shape(x, c_shape_ir) => isOfClass(x, l_Malignant)"""
    a = AXIOMS[name]
    def atom(pred, c):
        return f"{pred}(x, {c})" if isinstance(c, str) else "(" + " or ".join(f"{pred}(x, {x})" for x in c) + ")"
    parts = [atom(p, c) for p, c in a["body"]] + [f"not {atom(p, c)}" for p, c in a["nots"]]
    return f"{name}: forall x Diag(x, {a['guard']}): " + " and ".join(parts) + f" => isOfClass(x, {a['head']})"


def resolve_axioms(spec: Union[str, Sequence[str], None]) -> Tuple[str, ...]:
    """A preset of AXIOM_SETS, an axiom name, or a list of names; None / () / "" = no axioms."""
    if not spec:
        return ()
    if isinstance(spec, str):
        if spec in AXIOM_SETS:
            names = AXIOM_SETS[spec]
        elif spec in AXIOMS:
            names = (spec,)
        else:
            raise KeyError(f"axioms {spec!r}: not a preset {list(AXIOM_SETS)} nor an axiom {list(AXIOMS)}")
    else:
        names = tuple(spec)
    unknown = [n for n in names if n not in AXIOMS]
    if unknown:
        raise KeyError(f"unknown axioms {unknown}; known: {list(AXIOMS)}")
    normal = [n for n in names if n in NORMAL_AXIOMS]
    if normal:
        raise ValueError(f"axioms {normal} conclude isOfClass(x, l_Normal): the detector has no Normal class")
    if len(set(names)) != len(names):
        raise ValueError(f"axiom listed twice in {names}")
    return names


def axiom_heads(name: str) -> set:
    a = AXIOMS[name]
    return {PREDICATE_HEAD[p] for p, _ in a["body"] + a["nots"] if p in PREDICATE_HEAD}


def axiom_family(name: str) -> str:
    a = AXIOMS[name]
    fams = {"MASS" if c == "c_mass" else "CALC" for p, c in a["body"] if p in ("has_mass", "has_calc")}
    if len(fams) != 1:
        raise ValueError(f"{name}: needs exactly one of has_mass / has_calc in the body")
    return fams.pop()


def axiom_truths(name: str, cls, attr_probs, value_index, activation, family: str = "pred") -> torch.Tensor:
    """Truth of  body => isOfClass(x, head)  on every region.
    family="pred" (default, as in the paper): has_mass(x, c_mass) / has_calc(x, c_calc) predicted by the class head.
    family="label": the region's family is fixed by the domain (axiom_domain), so that atom is left out."""
    a = AXIOMS[name]
    parts = []
    for pred, c in a["body"]:
        if family == "label" and pred in ("has_mass", "has_calc"):
            continue
        parts.append(literal(pred, c, cls, attr_probs, value_index, activation))
    for pred, c in a["nots"]:
        parts.append(Not(literal(pred, c, cls, attr_probs, value_index, activation)))
    # antecedent = product of the body literals (negated ones as 1 - t); consequent = isOfClass(x, head)
    return Implies(And(*parts), cls[a["head"]])


def axiom_domain(name: str, labels: torch.Tensor, guard: str = "class", family: str = "pred"
                 ) -> Optional[torch.Tensor]:
    """The regions of forall x Diag(x, l_k): guard="class" = regions whose label is the class of the axiom
    (l_Benign: labels 1, 3; l_Malignant: 2, 4), as in the paper; "fg" = every lesion region; "all" = every region.
    family="label" also keeps only the regions of the axiom's family (our variant, see axiom_truths)."""
    if guard == "all":
        dom = None
    elif guard == "fg":
        dom = labels > 0
    elif guard == "class":
        # Diag(x, l_k) of the paper: only the regions whose ground-truth class is the axiom's class. Every foreground
        # region has a label, so on this domain the consequent is also trained by phi_class
        ks = CLASS_SETS[AXIOMS[name]["guard"]]
        dom = (labels == ks[0]) | (labels == ks[1])
    else:
        raise ValueError(f"ax_guard {guard!r}: use 'class', 'fg' or 'all'")
    if family == "pred":
        return dom
    if family != "label":
        raise ValueError(f"ax_family {family!r}: use 'pred' or 'label'")
    fk = FAMILY_SETS[axiom_family(name)]
    fam = (labels == fk[0]) | (labels == fk[1])
    return fam if dom is None else dom & fam


def axioms_sat(names: Sequence[str], cls, attr_probs, value_index, p: float, activation, labels: torch.Tensor,
               guard: str = "class", family: str = "pred"
               ) -> Tuple[Optional[torch.Tensor], Dict[str, Optional[torch.Tensor]], Dict[str, int]]:
    """Sat_axioms = SatAgg of the axioms; Sat per axiom (None = empty domain in this batch); domain sizes."""
    per, sizes = {}, {}
    for n in names:
        dom = axiom_domain(n, labels, guard, family)
        sizes[n] = int(labels.numel() if dom is None else dom.sum())
        # ApME over the axiom's domain; None when the domain is empty in this batch (vacuous: skipped, not counted)
        per[n] = apme(axiom_truths(n, cls, attr_probs, value_index, activation, family), p, mask=dom)
    return sat_agg(list(per.values()), p), per, sizes


# --------------------------------------------------------------------------------------------------
# Regularisers of SymMedNet "+Aux" (B.5), over the regions of the head's family; a sigmoid head is normalised to a
# distribution (q = p / sum p). Kept for the Ax+Aux variant; not in the plan of 6 Nov.
# --------------------------------------------------------------------------------------------------
AUX_HEADS = {"ent": ("mass_margin", "mass_shape", "calc_distr", "calc_morph"),
             "kl": ("calc_distr", "calc_morph"),
             "rec": ("mass_margin", "calc_distr", "calc_morph", "mass_shape", "mass_density")}


def _as_dist(p: torch.Tensor, activation: str) -> torch.Tensor:
    return p if activation == "softmax" else p / p.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def entropy_target_loss(q: torch.Tensor, target: float = 0.7) -> torch.Tensor:
    c = q.shape[-1]
    h = -(q * torch.log(q.clamp(min=1e-8))).sum(dim=-1)
    logc = torch.log(torch.tensor(float(c), device=q.device))
    return 1.0 - (1.0 - (h - target * logc).abs() / logc).mean()


def kl_uniform_loss(q: torch.Tensor) -> torch.Tensor:
    c = q.shape[-1]
    pbar = q.mean(dim=0).clamp(min=1e-8)
    return 1.0 - torch.exp(-(pbar * torch.log(pbar * c)).sum())


def head_domain(head: str, family: Dict[str, str], labels: torch.Tensor) -> torch.Tensor:
    ks = FAMILY_SETS[family[head]]
    return (labels == ks[0]) | (labels == ks[1])


def aux_losses(attr_probs, z, decoders, labels, activation, family, which, ent_target: float = 0.7,
               norm: str = "mean") -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """loss_ent, loss_kl, loss_rec. Reconstruction target: z standardised per region and detached; mean square."""
    losses, log = {}, {}
    n_total = labels.numel()
    for kind in which:
        terms = []
        for h in AUX_HEADS[kind]:
            dom = head_domain(h, family, labels)
            if not dom.any():
                continue
            if kind == "ent":
                v = entropy_target_loss(_as_dist(attr_probs[h][dom], activation[h]), ent_target)
            elif kind == "kl":
                v = kl_uniform_loss(_as_dist(attr_probs[h][dom], activation[h]))
            elif kind == "rec":
                v = F.mse_loss(decoders[h](attr_probs[h][dom]), F.layer_norm(z[dom].detach(), z.shape[-1:]))
            else:
                raise KeyError(f"aux {kind!r}: use ent, kl, rec")
            log[f"aux_{kind}/{h}"] = float(v.detach())
            terms.append(v * (float(dom.sum()) / n_total if norm == "region" else 1.0))
        if terms:
            losses[f"loss_{kind}"] = torch.stack(terms).sum()
    return losses, log


# --------------------------------------------------------------------------------------------------
# The whole knowledge base of one batch
# --------------------------------------------------------------------------------------------------
def _group_loss(per: Dict[str, Optional[torch.Tensor]], shares: Dict[str, float], p: float, norm: str
                ) -> Optional[torch.Tensor]:
    """1 - Sat of a group of formulas. norm="mean": SatAgg (SymMedNet). norm="region": mean over the formulas of
    |D| / N * (1 - Sat), D = the formula's domain, N = all sampled regions (as torchvision's box loss)."""
    if norm == "mean":
        s = sat_agg(list(per.values()), p)
        return None if s is None else 1.0 - s
    # |D| / N * (1 - Sat): a formula over 3 regions weighs less than one over 300, as a per-region loss would
    terms = [shares[k] * (1.0 - v) for k, v in per.items() if v is not None]
    return torch.stack(terms).sum() / len(per) if terms else None


def kb_losses(class_logits: torch.Tensor, labels: torch.Tensor, attr_logits: Dict[str, torch.Tensor],
              attr_targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]], value_index, p: float,
              activation: Dict[str, str], class_loss: str = "sat", class_agg: str = "geo",
              class_bg_weight: float = 0.5, use_attr_facts: bool = True, axioms: Sequence[str] = DEFAULT_AXIOMS,
              ax_guard: str = "class", ax_family: str = "pred", norm: str = "mean"
              ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Losses of one batch of sampled regions (all images concatenated):
        loss_class_sat  1 - Sat(phi_class)                     (class_loss="sat"; "ce" = torchvision's CE instead)
        loss_find       1 - Sat(phi1, phi2)                    (with the concept facts, as SymMedNet's L_find)
        loss_attr       1 - Sat(phi3 ... phi21)                (use_attr_facts)
        loss_clin       1 - Sat(psi)                           (axioms)
    and the satisfaction values for logging."""
    if norm not in ("mean", "region"):
        raise ValueError(f"norm {norm!r}: use 'mean' or 'region'")
    if class_loss not in ("sat", "ce"):
        raise ValueError(f"class_loss {class_loss!r}: use 'sat' or 'ce'")
    losses, log = {}, {}
    n = class_logits.shape[0]
    # truth values of all atoms on the R regions of the batch
    cls = class_predicates(class_logits)
    probs = head_probs(attr_logits, activation)

    # phi_class: always logged; it is the class loss only with class_loss='sat' (else torchvision's CE is used)
    s_class = class_facts_sat(class_logits, labels, p, class_agg, class_bg_weight)
    log["sat_class"] = float(s_class.detach())
    if class_loss == "sat":
        losses["loss_class_sat"] = 1.0 - s_class

    if use_attr_facts:
        # phi1, phi2 and phi3-phi21: the 'C' runs
        find = finding_facts_sat(cls, labels, p)
        # share of the sampled regions in each formula's domain (used by norm='region')
        shares = {"phi1_has_mass": float(((labels == MB) | (labels == MM)).sum()) / n,
                  "phi2_has_calc": float(((labels == CB) | (labels == CM)).sum()) / n}
        l = _group_loss(find, shares, p, norm)
        if l is not None:
            losses["loss_find"] = l
        log.update({f"sat_find/{k}": float(v.detach()) for k, v in find.items() if v is not None})
        if probs:
            per = {h: head_facts_sat(pr, attr_targets[h][0], attr_targets[h][1], p, activation[h])
                   for h, pr in probs.items()}
            shares = {h: float(attr_targets[h][1].sum()) / n for h in per}
            l = _group_loss(per, shares, p, norm)
            if l is not None:
                losses["loss_attr"] = l
            log.update({f"sat_attr/{h}": float(v.detach()) for h, v in per.items() if v is not None})
    if axioms:
        # psi: the 'Ax' runs; each axiom over its own domain, loss_clin = 1 - Sat of the group
        s, per, sizes = axioms_sat(axioms, cls, probs, value_index, p, activation, labels, ax_guard, ax_family)
        l = _group_loss(per, {k: v / n for k, v in sizes.items()}, p, norm)
        if l is not None:
            losses["loss_clin"] = l
        log.update({f"sat_ax/{k}": float(v.detach()) for k, v in per.items() if v is not None})
        if s is not None:
            log["sat_ax_all"] = float(s.detach())
    return losses, log
