"""
Tests of ltn_logic.py and ltn_attributes.py (version 5: SymMedNet knowledge base in the paper's notation, facts file). Run with:  python test_ltn.py   (no GPU, no data needed)
Each test states the property it checks; a failure stops with an AssertionError.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import ltn_attributes as A
import ltn_logic as L

torch.manual_seed(0)
T = lambda *v: torch.tensor(v, dtype=torch.float32)


def close(a, b, tol=1e-5):
    a, b = (x.detach() if torch.is_tensor(x) else x for x in (a, b))
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------------------------------------- connectives
def test_connectives_are_classical_on_0_1():
    for a in (0.0, 1.0):
        for b in (0.0, 1.0):
            ta, tb = T(a), T(b)
            assert close(L.And(ta, tb), a and b)
            assert close(L.Or(ta, tb), a or b)
            assert close(L.Implies(ta, tb), (not a) or b)
        assert close(L.Not(T(a)), 1 - a)


def test_implication_example_of_the_thesis():
    # Mass = 0.90, irregular = 0.80, Malignant = 0.30  ->  psi6 = 1 - 0.72 + 0.72*0.30 = 0.496
    assert close(L.Implies(L.And(T(0.9), T(0.8)), T(0.3)), 0.496)


# ---------------------------------------------------------------------------------------------- quantifier
def test_apme_p1_is_the_mean():
    t = T(0.2, 0.5, 0.9)
    assert close(L.apme(t, 1.0), (1 - L.EPS) * t.mean(), 1e-6)


def test_apme_large_p_approaches_the_minimum():
    t = T(0.95, 0.95, 0.95, 0.2)
    assert L.apme(t, 1.0) > L.apme(t, 4.0) > L.apme(t, 10.0) > L.apme(t, 50.0)
    assert L.apme(t, 50.0) < 0.35


def test_apme_satisfied_formula_has_finite_gradient():
    t = torch.ones(5, requires_grad=True)
    s = L.apme(t, 2.0)
    s.backward()
    assert s > 0.99 and torch.isfinite(t.grad).all()


def test_apme_empty_domain_is_vacuous():
    assert L.apme(T(0.3, 0.4), 2.0, mask=torch.tensor([False, False])) is None
    assert L.sat_agg([None, None], 2.0) is None
    assert close(L.sat_agg([None, T(0.7)[0]], 2.0), (1 - L.EPS) * 0.7, 1e-6)


def test_p_schedule_follows_symmednet():
    assert [L.p_schedule(e, 150) for e in (0, 49, 50, 89, 90, 119, 120, 139, 140, 149)] == \
           [2, 2, 4, 4, 6, 6, 8, 8, 10, 10]
    ps = [L.p_schedule(e, 50) for e in range(50)]
    assert ps[0] == 2 and ps[-1] == 10 and ps == sorted(ps)


# ---------------------------------------------------------------------------------------------- predicates, facts
def test_class_facts_ce_is_the_baseline_cross_entropy():
    logits = torch.randn(64, 5)
    labels = torch.randint(0, 5, (64,))
    assert close(L.class_fact_loss(logits, labels), F.cross_entropy(logits, labels), 1e-7)


def test_class_facts_sat_is_apme_of_the_labelled_class():
    logits = torch.randn(64, 5)
    labels = torch.randint(0, 5, (64,))
    t = F.softmax(logits, -1)[torch.arange(64), labels]
    assert close(L.class_facts_sat(logits, labels, 2.0, "all"), L.apme(t, 2.0), 1e-6)
    per = [L.apme(t, 2.0, mask=labels == k) for k in range(5)]
    assert close(L.class_facts_sat(logits, labels, 2.0, "per_class"), L.sat_agg(per, 2.0), 1e-6)
    sure = torch.full((4, 5), -20.0); sure[torch.arange(4), torch.tensor([0, 1, 2, 4])] = 20
    assert L.class_facts_sat(sure, torch.tensor([0, 1, 2, 4]), 2.0, "all") > 0.999, "correct and sure: Sat = 1"


def test_class_facts_geo_is_one_minus_exp_minus_ce():
    logits = torch.randn(64, 5)
    labels = torch.randint(0, 5, (64,))
    s = L.class_facts_sat(logits, labels, 2.0, "geo")
    assert close(1 - s, 1 - torch.exp(-F.cross_entropy(logits, labels)), 1e-6)


def test_geo_keeps_the_gradient_on_a_badly_classified_lesion():
    # 99 background regions classified well, one malignant calcification with p = 0.001: with ApME ("all") its
    # logit gradient is tiny, with the geometric mean it is ~ Sat x the cross-entropy gradient
    def grad(agg):
        logits = torch.full((100, 5), -4.0); logits[:, 0] = 4.0
        logits[99] = torch.tensor([6.0, 0.0, 0.0, 0.0, -1.0])
        logits.requires_grad_(True)
        labels = torch.zeros(100, dtype=torch.long); labels[99] = 4
        (1 - L.class_facts_sat(logits, labels, 2.0, agg)).backward()
        return float(-logits.grad[99, 4])
    g_all, g_geo = grad("all"), grad("geo")
    assert g_geo > 20 * g_all, (g_all, g_geo)


def test_balanced_weights_background_and_lesions():
    logits = torch.randn(50, 5)
    labels = torch.tensor([0] * 40 + [1] * 3 + [2] * 3 + [3] * 2 + [4] * 2)
    t = F.softmax(logits, -1)[torch.arange(50), labels]
    s = [L.apme(t, 2.0, mask=labels == k) for k in range(5)]
    w = [0.5] + [0.125] * 4
    want = 1 - (sum(wk * (1 - sk) ** 2 for wk, sk in zip(w, s)) / sum(w)) ** 0.5
    assert close(L.class_facts_sat(logits, labels, 2.0, "balanced", 0.5), want, 1e-6)
    only_bg = torch.zeros(10, dtype=torch.long)                     # absent classes are left out (weights renormalised)
    assert close(L.class_facts_sat(logits[:10], only_bg, 2.0, "balanced"), L.apme(t[:0].new_tensor(
        F.softmax(logits[:10], -1)[:, 0]), 2.0), 1e-6)
    for bad in (0.0, 1.0):
        try:
            L.class_facts_sat(logits, labels, 2.0, "balanced", bad); raise RuntimeError("should have raised")
        except ValueError:
            pass


def test_class_facts_sat_gradient_raises_the_labelled_class():
    logits = torch.zeros(3, 5, requires_grad=True)
    labels = torch.tensor([0, 2, 3])
    (1 - L.class_facts_sat(logits, labels, 2.0)).backward()
    for i, k in enumerate(labels.tolist()):
        assert logits.grad[i, k] < 0 and (logits.grad[i, [j for j in range(5) if j != k]] > 0).all()


def test_class_and_finding_predicates_are_sums_of_the_softmax():
    c = L.class_predicates(torch.randn(10, 5))
    p = c["probs"]
    assert torch.allclose(c["l_Malignant"], p[:, 2] + p[:, 4]) and torch.allclose(c["l_Benign"], p[:, 1] + p[:, 3])
    assert torch.allclose(c["c_mass"], p[:, 1] + p[:, 2]) and torch.allclose(c["c_calc"], p[:, 3] + p[:, 4])
    assert torch.allclose(c["c_mass"] + c["c_calc"] + p[:, 0], torch.ones(10), atol=1e-6)


def test_softmax_compound_is_a_disjunction():
    probs = T(0.1, 0.0, 0.0, 0.4, 0.5)[None]                    # distribution head of one region
    y = torch.zeros(1, 5); y[0, 3] = 1; y[0, 4] = 1               # equal suspicion kept as "A or B"
    assert close(L.attribute_fact_truths(probs, y)[0], 0.9)


def test_sigmoid_facts_are_one_per_value_with_negatives():
    probs = T(0.9, 0.1, 0.7, 0.2, 0.3)[None]                    # morphology of one cluster (sigmoid)
    y = T(1, 0, 1, 0, 0)[None]                                  # typically benign + pleomorphic, nothing else
    t = L.attribute_fact_truths(probs, y, "sigmoid")
    assert t.shape == (1, 5) and torch.allclose(t, T(0.9, 0.9, 0.7, 0.8, 0.7)[None])
    s = L.head_facts_sat(probs, y, torch.tensor([True]), 1.0, "sigmoid")
    assert close(s, (1 - L.EPS) * t.mean(), 1e-6), "p = 1: the plain mean over the five value-facts"
    assert L.head_facts_sat(probs, y, torch.tensor([False]), 2.0, "sigmoid") is None



def test_heads_of_version_5():
    assert A.VERSION == 5 and L.VERSION == 5
    assert A.ACTIVATION == {"mass_shape": "softmax", "mass_margin": "sigmoid", "mass_density": "softmax",
                            "calc_morph": "sigmoid", "calc_distr": "softmax"}, "as SymMedNet 3.3 and the call of 29 Oct"
    assert A.HEAD_SIZES["mass_density"] == 2 and "category" not in A.HEADS
    z = {h: torch.randn(6, k) for h, k in A.HEAD_SIZES.items()}
    pr = L.head_probs(z, A.ACTIVATION)
    for h in A.HEAD_NAMES:
        if A.ACTIVATION[h] == "softmax":
            assert torch.allclose(pr[h].sum(-1), torch.ones(6), atol=1e-6), h
        else:
            assert torch.allclose(pr[h], torch.sigmoid(z[h])), h


# ---------------------------------------------------------------------------------------------- axioms (paper notation)
VI, ACT = A.VALUE_INDEX, A.ACTIVATION


def _uniform(n=1):
    return L.head_probs({h: torch.zeros(n, k) for h, k in A.HEAD_SIZES.items()}, ACT)


def _cls(*probs):
    return L.class_predicates(T(*probs).clamp(min=1e-12).log()[None])


def test_axioms_are_written_as_in_the_paper():
    assert L.axiom_text("psi6") == ("psi6: forall x Diag(x, l_Malignant): has_mass(x, c_mass) and "
                                    "has_shape(x, c_shape_ir) => isOfClass(x, l_Malignant)")
    assert L.axiom_text("psi3") == ("psi3: forall x Diag(x, l_Benign): has_mass(x, c_mass) and "
                                    "(has_margin(x, c_margin_obs) or has_margin(x, c_margin_circ)) => isOfClass(x, l_Benign)")
    assert "not has_distribution(x, c_distr_reg) and not has_distribution(x, c_distr_dif)" in L.axiom_text("psi7")
    for n, a in L.AXIOMS.items():
        assert a["guard"] == a["head"], f"{n}: Diag(x, l_k) must use the class of isOfClass(x, l_k)"
        for pred, c in a["body"] + a["nots"]:
            for x in ([c] if isinstance(c, str) else c):
                assert x in L.CONSTANTS or x in L.FINDING_SETS, (n, x)


def test_the_sixteen_axioms_of_symmednet_without_normal():
    assert L.DEFAULT_AXIOMS == tuple(f"psi{i}" for i in range(3, 15)) + ("psi16",) and len(L.DEFAULT_AXIOMS) == 13
    assert set(L.NORMAL_AXIOMS) == {"psi1", "psi2", "psi15"}
    assert L.resolve_axioms("kb16") == L.DEFAULT_AXIOMS and L.resolve_axioms(None) == () and L.resolve_axioms("") == ()
    assert L.resolve_axioms("kb14") == tuple(f"psi{i}" for i in range(3, 15))
    for bad, err in ((["B1"], KeyError), ("kb99", KeyError), (["psi15"], ValueError), ("psi1", ValueError),
                     (["psi6", "psi6"], ValueError)):
        try:
            L.resolve_axioms(bad); raise RuntimeError(f"should have raised: {bad}")
        except err:
            pass


def test_every_axiom_is_well_formed():
    cls = L.class_predicates(torch.randn(7, 5))
    cls["l_Normal"] = torch.rand(7)                                  # only to evaluate psi1, psi2, psi15 here
    attr = L.head_probs({h: torch.randn(7, k) for h, k in A.HEAD_SIZES.items()}, ACT)
    for n in L.AXIOMS:
        for fam in ("pred", "label"):
            t = L.axiom_truths(n, cls, attr, VI, ACT, fam)
            assert t.shape == (7,) and (t >= 0).all() and (t <= 1 + 1e-6).all(), n
        assert L.axiom_heads(n) <= set(A.HEAD_SIZES), n
        assert L.axiom_family(n) in ("MASS", "CALC")


def test_a_constant_of_another_predicate_is_refused():
    try:
        L.literal("has_shape", "c_margin_spic", _cls(0, 1, 0, 0, 0), _uniform(), VI, ACT); raise RuntimeError("raise")
    except KeyError:
        pass


def test_psi6_by_hand():
    # has_mass 0.9, has_shape(c_shape_ir) 0.8, isOfClass(l_Malignant) 0.3 -> 1 - 0.72 + 0.72 * 0.3 = 0.496
    cls = _cls(0.1, 0.6, 0.3, 0, 0)
    attr = _uniform(); attr["mass_shape"] = T(0.1, 0.1, 0.8)[None]
    assert close(L.axiom_truths("psi6", cls, attr, VI, ACT)[0], 0.496, 1e-5)
    # family="label": has_mass is fixed by the domain -> 1 - 0.8 + 0.8 * 0.3 = 0.44
    assert close(L.axiom_truths("psi6", cls, attr, VI, ACT, "label")[0], 0.44, 1e-5)


def test_psi3_margin_is_a_sigmoid_disjunction():
    cls = _cls(0, 0, 1, 0, 0)                                      # a malignant mass: l_Benign = 0
    attr = _uniform(); attr["mass_margin"] = T(0.5, 0.5, 0.0, 0.0, 0.0)[None]
    assert close(L.axiom_truths("psi3", cls, attr, VI, ACT)[0], 0.25, 1e-5)   # 1 - 1 * 0.75 + 0


def test_psi7_excludes_regional_and_diffuse():
    cls = _cls(0, 0, 0, 1, 0)                                      # a benign calcification
    attr = _uniform(); attr["calc_morph"] = T(0, 0, 0, 1, 0)[None]; attr["calc_distr"] = T(0, 0, 0, 1, 0)[None]
    assert L.axiom_truths("psi7", cls, attr, VI, ACT)[0] > 0.99, "amorphous but regional: psi7 says nothing"
    attr["calc_distr"] = T(1, 0, 0, 0, 0)[None]
    assert L.axiom_truths("psi7", cls, attr, VI, ACT)[0] < 0.01


def test_class_guard_quantifies_each_axiom_over_its_class():
    labels = torch.tensor([0, 1, 2, 3, 4, 2])
    assert L.axiom_domain("psi6", labels).tolist() == [False, False, True, False, True, True], "Diag(x, l_Malignant)"
    assert L.axiom_domain("psi3", labels).tolist() == [False, True, False, True, False, False], "Diag(x, l_Benign)"
    assert L.axiom_domain("psi6", labels, family="label").tolist() == [False, False, True, False, False, True]
    assert L.axiom_domain("psi6", labels, "fg").tolist() == [False, True, True, True, True, True]
    assert L.axiom_domain("psi6", labels, "all") is None


def test_class_guarded_axiom_never_asks_against_the_label():
    cls = L.class_predicates(T(-9, 9, -9, -9, -9)[None])             # a benign irregular mass, predicted benign
    attr = _uniform(); attr["mass_shape"] = T(0, 0, 1)[None]
    labels = torch.tensor([1])
    _, per_all, _ = L.axioms_sat(["psi6"], cls, attr, VI, 2.0, ACT, labels, "all")
    _, per_cls, sizes = L.axioms_sat(["psi6"], cls, attr, VI, 2.0, ACT, labels, "class")
    assert per_all["psi6"] < 0.01 and per_cls["psi6"] is None and sizes["psi6"] == 0


def _psi6_class_grad(family):
    logits = torch.zeros(1, 5, requires_grad=True)
    attr = _uniform(); attr["mass_shape"] = T(0, 0, 1)[None]
    s, _, _ = L.axioms_sat(["psi6"], L.class_predicates(logits), attr, VI, 2.0, ACT, torch.tensor([2]), "class", family)
    (1 - s).backward()
    return logits.grad[0]


def test_axiom_raises_malignant_on_an_irregular_malignant_mass():
    g = _psi6_class_grad("label")
    assert g[2] < 0 and g[4] < 0 and g[1] > 0 and g[3] > 0, "isOfClass(l_Malignant) up, l_Benign down"


def test_predicted_family_as_in_the_paper_can_move_mass_to_calc():
    # documented property of family="pred" (the default, as the paper): on a malignant mass psi6 can also be
    # satisfied by lowering has_mass, i.e. moving probability from "malignant mass" to "malignant calcification"
    g = _psi6_class_grad("pred")
    assert g[2] > 0 and g[4] < 0


def test_facts_outweigh_the_shortcut_on_an_annotated_concept():
    cls = _cls(0.0, 0.7, 0.3, 0, 0)
    shape_logits = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    attr = _uniform(); attr["mass_shape"] = F.softmax(shape_logits, -1)
    s_clin, _, _ = L.axioms_sat(["psi6"], cls, attr, VI, 2.0, ACT, torch.tensor([2]))
    g_short = torch.autograd.grad(1 - s_clin, shape_logits, retain_graph=True)[0][0, 2]
    assert g_short > 0, "axiom alone pushes irregular DOWN (the shortcut exists)"
    s_f = L.head_facts_sat(attr["mass_shape"], T(0, 0, 1)[None], torch.tensor([True]), 2.0, "softmax")
    g_tot = torch.autograd.grad((1 - s_clin) + (1 - s_f), shape_logits)[0][0, 2]
    assert g_tot < 0, "with the fact, the net gradient pushes irregular UP"


# ---------------------------------------------------------------------------------------------- finding facts
def test_phi1_phi2_over_the_mass_and_calcification_regions():
    logits = torch.randn(6, 5)
    labels = torch.tensor([0, 1, 2, 3, 4, 0])
    cls = L.class_predicates(logits)
    f = L.finding_facts_sat(cls, labels, 2.0)
    assert close(f["phi1_has_mass"], L.apme(cls["c_mass"][1:3], 2.0), 1e-6)
    assert close(f["phi2_has_calc"], L.apme(cls["c_calc"][3:5], 2.0), 1e-6)
    assert L.finding_facts_sat(cls, torch.zeros(6, dtype=torch.long), 2.0)["phi1_has_mass"] is None


# ---------------------------------------------------------------------------------------------- aux regularisers
def test_entropy_target_values():
    u, one = torch.full((4, 5), 0.2), torch.eye(5)[:4]
    assert close(L.entropy_target_loss(u, 0.7), 0.3, 1e-5), "uniform: H = log C, s = 0.7"
    assert close(L.entropy_target_loss(one, 0.7), 0.7, 1e-5), "one-hot: H = 0, s = 0.3"


def test_kl_uniform_values():
    assert close(L.kl_uniform_loss(torch.eye(5)), 0.0, 1e-5), "batch marginal uniform: no penalty"
    collapsed = torch.zeros(6, 5); collapsed[:, 2] = 1
    assert close(L.kl_uniform_loss(collapsed), 1 - 1 / 5, 1e-4), "all on one value: 1 - 1/C"


def test_aux_losses_use_the_family_regions_and_backpropagate():
    n = 12
    labels = torch.tensor([0, 0, 1, 2, 2, 1, 3, 4, 3, 0, 0, 0])
    z = torch.randn(n, 16, requires_grad=True)
    logits = {h: torch.randn(n, k, requires_grad=True) for h, k in A.HEAD_SIZES.items()}
    probs = L.head_probs(logits, ACT)
    dec = torch.nn.ModuleDict({h: torch.nn.Linear(A.HEAD_SIZES[h], 16) for h in L.AUX_HEADS["rec"]})
    losses, log = L.aux_losses(probs, z, dec, labels, ACT, A.FAMILY, ("ent", "kl", "rec"))
    assert set(losses) == {"loss_ent", "loss_kl", "loss_rec"}
    sum(losses.values()).backward()
    for h in ("mass_shape", "calc_morph"):
        assert logits[h].grad[labels == 0].abs().sum() == 0, "background regions untouched"
    assert logits["mass_shape"].grad[torch.isin(labels, torch.tensor([3, 4]))].abs().sum() == 0, "mass head off calcs"
    assert logits["calc_distr"].grad[torch.isin(labels, torch.tensor([3, 4]))].abs().sum() > 0
    mass = torch.isin(labels, torch.tensor([1, 2]))
    want = L.entropy_target_loss(probs["mass_shape"][mass], 0.7)
    assert close(log["aux_ent/mass_shape"], want, 1e-6)
    zero = torch.nn.ModuleDict({h: torch.nn.Linear(A.HEAD_SIZES[h], 16) for h in L.AUX_HEADS["rec"]})
    for m in zero.values():
        torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
    big = (z * 100).detach()
    rec, _ = L.aux_losses(probs, big, zero, labels, ACT, A.FAMILY, ("rec",))
    assert close(rec["loss_rec"], len(L.AUX_HEADS["rec"]) * 1.0, 1e-3), "standardised target: a zero decoder scores 1"
    reg, _ = L.aux_losses(probs, z, dec, labels, ACT, A.FAMILY, ("ent",), norm="region")
    want = 0.0
    for h in L.AUX_HEADS["ent"]:
        dom = L.head_domain(h, A.FAMILY, labels)
        want += float(L.entropy_target_loss(L._as_dist(probs[h][dom], ACT[h]), 0.7).detach()) * float(dom.sum()) / n
    assert close(reg["loss_ent"], want, 1e-5), "region: each head weighs |D_h| / N (4 masses, 3 calcifications)"


# ---------------------------------------------------------------------------------------------- whole KB
def _random_targets(n, frac=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    targets = {}
    for h, k in A.HEAD_SIZES.items():
        if A.ACTIVATION[h] == "softmax":
            y = torch.zeros(n, k); y[torch.arange(n), torch.randint(0, k, (n,), generator=g)] = 1
        else:
            y = (torch.rand(n, k, generator=g) < 0.4).float()
        targets[h] = (y, torch.rand(n, generator=g) < frac)
    return targets


# ---------------------------------------------------------------------------------------------- whole KB
def _random_targets(n, frac=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    targets = {}
    for h, k in A.HEAD_SIZES.items():
        if A.ACTIVATION[h] == "softmax":
            y = torch.zeros(n, k); y[torch.arange(n), torch.randint(0, k, (n,), generator=g)] = 1
        else:
            y = (torch.rand(n, k, generator=g) < 0.4).float()
        targets[h] = (y, torch.rand(n, generator=g) < frac)
    return targets


def test_kb_losses_run_and_backpropagate():
    n = 40
    cls_logits = torch.randn(n, 5, requires_grad=True)
    labels = torch.randint(0, 5, (n,))
    attr_logits = {h: torch.randn(n, k, requires_grad=True) for h, k in A.HEAD_SIZES.items()}
    losses, log = L.kb_losses(cls_logits, labels, attr_logits, _random_targets(n), VI, p=4.0, activation=ACT,
                              axioms=L.resolve_axioms("kb16"))
    assert set(losses) == {"loss_class_sat", "loss_find", "loss_attr", "loss_clin"}
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(cls_logits.grad).all()
    assert all(torch.isfinite(v.grad).all() for v in attr_logits.values())
    assert "sat_class" in log and "sat_ax_all" in log and "sat_find/phi1_has_mass" in log


def test_the_four_steps_of_the_plan_have_the_right_terms():
    n = 30
    logits, labels = torch.randn(n, 5), torch.randint(0, 5, (n,))
    attr_logits = {h: torch.randn(n, k) for h, k in A.HEAD_SIZES.items()}
    t = _random_targets(n)
    kw = dict(value_index=VI, p=2.0, activation=ACT)
    s0, _ = L.kb_losses(logits, labels, {}, {}, use_attr_facts=False, axioms=(), **kw)
    c, _ = L.kb_losses(logits, labels, attr_logits, t, use_attr_facts=True, axioms=(), **kw)
    axc, _ = L.kb_losses(logits, labels, attr_logits, t, use_attr_facts=True, **kw)
    ax, _ = L.kb_losses(logits, labels, attr_logits, {}, use_attr_facts=False, **kw)
    assert set(s0) == {"loss_class_sat"}, "S0: the baseline converted to LTN"
    assert set(c) == {"loss_class_sat", "loss_find", "loss_attr"}, "C: + facts"
    assert set(axc) == {"loss_class_sat", "loss_find", "loss_attr", "loss_clin"}, "AxC: + facts + axioms"
    assert set(ax) == {"loss_class_sat", "loss_clin"}, "Ax: axioms only, no concept labels needed"
    ce, log = L.kb_losses(logits, labels, {}, {}, class_loss="ce", use_attr_facts=False, axioms=(), **kw)
    assert ce == {} and "sat_class" in log, "class_loss='ce': torchvision's CE is used, Sat only logged"


def test_ce_mode_uses_bce_on_sigmoid_heads():
    n = 12
    z = {"calc_morph": torch.randn(n, 5), "mass_margin": torch.randn(n, 5)}
    t = _random_targets(n, frac=0.5)
    loss, log = L.attribute_ce_loss(z, t, ACT)
    for h in z:
        y, v = t[h]
        assert close(log[f"ce_attr/{h}"], F.binary_cross_entropy_with_logits(z[h][v], y[v]), 1e-5), h


def test_region_norm_scales_every_term_by_its_domain():
    n = 40
    cls_logits = torch.randn(n, 5)
    labels = torch.zeros(n, dtype=torch.long); labels[:8] = 2; labels[8:13] = 1
    attr_logits = {h: torch.randn(n, k) for h, k in A.HEAD_SIZES.items()}
    targets = {}
    for h, k in A.HEAD_SIZES.items():
        y = torch.zeros(n, k); y[:, 0] = 1
        v = torch.zeros(n, dtype=torch.bool)
        if h == "mass_margin":
            v[:8] = True
        targets[h] = (y, v)
    losses, _ = L.kb_losses(cls_logits, labels, attr_logits, targets, VI, p=2.0, activation=ACT,
                            axioms=("psi6", "psi3"), norm="region")
    probs = L.head_probs(attr_logits, ACT)
    _, per = L.attribute_facts_sat(probs, targets, 2.0, ACT)
    assert close(losses["loss_attr"], (8 / n) * (1 - per["mass_margin"]) / len(A.HEAD_NAMES), 1e-6)
    cls = L.class_predicates(cls_logits)
    _, pax, sizes = L.axioms_sat(("psi6", "psi3"), cls, probs, VI, 2.0, ACT, labels)
    assert sizes == {"psi6": 8, "psi3": 5}
    assert close(losses["loss_clin"], ((8 / n) * (1 - pax["psi6"]) + (5 / n) * (1 - pax["psi3"])) / 2, 1e-6)
    f = L.finding_facts_sat(cls, labels, 2.0)
    assert close(losses["loss_find"], (13 / n) * (1 - f["phi1_has_mass"]) / 2, 1e-6), "no calcification: phi2 vacuous"
    assert close(losses["loss_class_sat"], 1 - L.class_facts_sat(cls_logits, labels, 2.0), 1e-6), "class: weight 1"


def test_region_norm_ce_is_the_mean_times_the_annotated_share():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(200, 5, generator=g)
    y = torch.zeros(200, 5); y[torch.arange(200), torch.randint(0, 5, (200,), generator=g)] = 1
    valid = torch.zeros(200, dtype=torch.bool); valid[:10] = True
    t = {"calc_distr": (y, valid)}
    mean, _ = L.attribute_ce_loss({"calc_distr": z}, t, ACT)
    region, log = L.attribute_ce_loss({"calc_distr": z}, t, ACT, n_total=200)
    assert close(region, mean * 10 / 200, 1e-6) and close(log["ce_attr/calc_distr"], mean, 1e-6)


# ---------------------------------------------------------------------------------------------- concepts (CBIS)
def _mass(shape, margin):
    return {"family": "MASS", "mass shape": shape, "mass margins": margin}


def _calc(ctype, distr):
    return {"family": "CALC", "calc type": ctype, "calc distribution": distr}


def test_margin_lists_every_value_and_shape_keeps_the_most_suspicious():
    a = A.csv_attributes(_mass("LOBULATED-IRREGULAR", "CIRCUMSCRIBED-SPICULATED"))
    assert a["mass_shape"] == [0, 0, 1] and a["mass_margin"] == [1, 0, 0, 0, 1], "margin sigmoid: both listed"
    a = A.csv_attributes(_mass("ROUND-OVAL", "CIRCUMSCRIBED-OBSCURED"))
    assert a["mass_shape"] == [1, 1, 0] and a["mass_margin"] == [1, 1, 0, 0, 0]
    a = A.csv_attributes(_mass("LOBULATED", "ILL_DEFINED"))
    assert a["mass_shape"] == [1, 0, 0] and a["mass_margin"] == [0, 0, 0, 1, 0], "lobulated -> oval, ill defined -> indistinct"
    a = A.csv_attributes(_mass("ARCHITECTURAL_DISTORTION", float("nan")))
    assert a["mass_shape"] is None and a["mass_margin"] is None, "not a shape value / missing: no fact"
    assert "category" not in a


def test_morphology_lists_every_value_and_distribution_the_most_suspicious():
    a = A.csv_attributes(_calc("PUNCTATE-PLEOMORPHIC", "CLUSTERED-LINEAR"))
    assert a["calc_morph"] == [1, 0, 1, 0, 0] and a["calc_distr"] == [0, 0, 1, 0, 0]
    a = A.csv_attributes(_calc("LUCENT_CENTERED", "DIFFUSELY_SCATTERED"))
    assert a["calc_morph"] == [1, 0, 0, 0, 0] and a["calc_distr"] == [0, 0, 0, 0, 1]
    a = A.csv_attributes(_calc("FINE_LINEAR_BRANCHING", "LINEAR-SEGMENTAL"))
    assert a["calc_distr"] == [0, 1, 0, 0, 0]


def test_unknown_tokens_raise():
    for bad in (_mass("OVALISH", "SPICULATED"), _calc("PLEOMORPHIC", "SCATTERED")):
        try:
            A.csv_attributes(bad); raise RuntimeError("should have raised")
        except KeyError:
            pass


def test_density_is_a_two_value_softmax_fact_on_single_lesion_images():
    row = {"mass_density_HIGH": 1, "mass_density_LOW": 0}
    la = A.lesion_attributes(_mass("OVAL", "SPICULATED"), row, "MASS", 1)
    assert la["mass_density"] == [1.0, 0.0]
    assert A.lesion_attributes(_mass("OVAL", "SPICULATED"), row, "MASS", 2)["mass_density"] is None
    assert A.lesion_attributes(None, {"mass_density_HIGH": 0, "mass_density_LOW": 1}, "MASS", 1) == \
        {"mass_density": [0.0, 1.0]}, "no CSV row: only density"
    assert A.lesion_attributes(None, {"mass_density_HIGH": 1, "mass_density_LOW": 1}, "MASS", 1) is None
    assert A.lesion_attributes(None, row, "CALC", 1) is None
    try:
        A.lesion_attributes(_calc("AMORPHOUS", "CLUSTERED"), row, "MASS", 1); raise RuntimeError("should have raised")
    except ValueError:
        pass


def test_load_cbis_csvs_drops_rows_outside_train_val():
    import tempfile, os
    import pandas as pd
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mass.csv")
    pd.DataFrame({"image file path": ["Mass-Training_P_1_LEFT_CC/1.2/3.dcm", "Mass-Test_P_9_LEFT_CC/1/2.dcm"],
                  "abnormality id": [1, 1], "abnormality type": ["mass", "mass"], "assessment": [4, 5],
                  "mass shape": ["OVAL", "ROUND"], "mass margins": ["SPICULATED", "OBSCURED"]}).to_csv(p, index=False)
    idx, kept = A.load_cbis_csvs([p], {"Mass-Training_P_1_LEFT_CC"})
    assert list(idx) == [("Mass-Training_P_1_LEFT_CC", 1)] and kept == {"mass.csv": 1}


def test_target_tensors_follow_the_box_order_and_refuse_old_records():
    m = A.lesion_attributes(_mass("OVAL", "SPICULATED"), {"mass_density_HIGH": 0, "mass_density_LOW": 1}, "MASS", 1)
    c = A.lesion_attributes(_calc("PLEOMORPHIC", "CLUSTERED"), {}, "CALC", 1)
    t = A.target_tensors([m, None, c])
    assert t["attr_mass_shape_valid"].tolist() == [True, False, False]
    assert t["attr_calc_morph_valid"].tolist() == [False, False, True]
    assert t["attr_calc_morph"][2].tolist() == [0, 0, 1, 0, 0]
    assert t["attr_mass_density"][0].tolist() == [0, 1] and "attr_category" not in t
    try:
        A.target_tensors([{"mass_density": [1.0]}]); raise RuntimeError("should have raised")
    except ValueError:
        pass



# ---------------------------------------------------------------------------------------------- facts file
def _facts_csv(rows, path):
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _two_lesions():
    mass = A.facts_row("Mass-Training_P_00001_LEFT_CC", 1, "train", "MASS", "MALIGNANT", 1,
                       {"family": "MASS", "mass shape": "IRREGULAR", "mass margins": "ILL_DEFINED-SPICULATED"},
                       {"mass_density_HIGH": 1, "mass_density_LOW": 0})
    calc = A.facts_row("Calc-Training_P_00002_LEFT_CC", 1, "val", "CALC", "BENIGN", 1,
                       {"family": "CALC", "calc type": "PUNCTATE-PLEOMORPHIC", "calc distribution": "CLUSTERED"}, {})
    return mass, calc


def test_facts_row_has_one_column_per_fact_of_the_paper():
    mass, calc = _two_lesions()
    assert mass["phi01_has_mass"] == 1 and mass["phi02_has_calc"] == 0
    assert mass["phi06_shape_irregular"] == 1 and mass["phi04_shape_oval"] == 0
    assert (mass["phi10_margin_indistinct"], mass["phi11_margin_spiculated"], mass["phi07_margin_circumscribed"]) == (1, 1, 0)
    assert mass["phi03_density_high"] == 1 and mass["phi03_density_low"] == 0
    assert mass["phi12_morph_typically_benign"] == "" and mass["phi17_distr_clustered"] == "", "other family: unknown"
    assert calc["phi12_morph_typically_benign"] == 1 and calc["phi14_morph_pleomorphic"] == 1
    assert calc["phi15_morph_amorphous"] == 0 and calc["phi17_distr_clustered"] == 1
    assert len([c for c in mass if c.startswith("phi")]) == 22, "phi1, phi2, phi3 (high + low), phi4-phi21"


def test_facts_file_round_trip():
    import tempfile, os
    mass, calc = _two_lesions()
    p = _facts_csv([mass, calc], os.path.join(tempfile.mkdtemp(), "facts.csv"))
    f = A.load_facts_file(p)
    m = f[("Mass-Training_P_00001_LEFT_CC", 1)]
    assert m["family"] == "MASS" and m["split"] == "train"
    assert m["attrs"]["mass_margin"] == [0, 0, 0, 1, 1] and m["attrs"]["mass_density"] == [1, 0]
    assert m["attrs"]["calc_morph"] is None
    c = f[("Calc-Training_P_00002_LEFT_CC", 1)]
    assert c["attrs"]["calc_morph"] == [1, 0, 1, 0, 0] and c["attrs"]["calc_distr"] == [1, 0, 0, 0, 0]
    t = A.target_tensors([m["attrs"], c["attrs"]])
    assert t["attr_mass_margin_valid"].tolist() == [True, False] and t["attr_calc_distr"][1].tolist() == [1, 0, 0, 0, 0]


def test_facts_file_refuses_test_rows_and_bad_cells():
    import tempfile, os
    mass, calc = _two_lesions()
    d = tempfile.mkdtemp()
    for change, err in ((dict(split="test"), ValueError), (dict(phi06_shape_irregular=2), ValueError),
                        (dict(phi04_shape_oval=0, phi05_shape_round=0, phi06_shape_irregular=0), ValueError),
                        (dict(family="CALC"), ValueError)):
        p = _facts_csv([{**mass, **change}, calc], os.path.join(d, "f.csv"))
        try:
            A.load_facts_file(p); raise RuntimeError(f"should have raised: {change}")
        except err:
            pass
    p = _facts_csv([mass, mass], os.path.join(d, "dup.csv"))
    try:
        A.load_facts_file(p); raise RuntimeError("should have raised: duplicate")
    except KeyError:
        pass
    try:
        A.facts_row("Mass-Test_P_1_LEFT_CC", 1, "test", "MASS", "BENIGN", 1, None, {}); raise RuntimeError("raise")
    except ValueError:
        pass



if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print(f"\n{len(tests)} tests passed")
