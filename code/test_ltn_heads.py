"""
Tests of ltn_heads.py (version 5) on a real torchvision Faster R-CNN (random weights, CPU, two small images).
Run with:  python test_ltn_heads.py      (about a minute on CPU)
"""
import copy

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import ltn_attributes as A
import ltn_heads as H
import ltn_logic as L

torch.manual_seed(0)


def tiny_model():
    m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=256, max_size=256)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 5)
    return m


def batch(with_attrs=True):
    imgs = [torch.rand(3, 256, 256), torch.rand(3, 256, 200)]
    t0 = {"boxes": torch.tensor([[40., 40, 120, 130], [150, 30, 190, 70]]), "labels": torch.tensor([2, 3])}
    t1 = {"boxes": torch.tensor([[20., 60, 90, 140]]), "labels": torch.tensor([1])}
    if with_attrs:
        mass = {"family": "MASS", "mass shape": "IRREGULAR", "mass margins": "SPICULATED"}
        calc = {"family": "CALC", "calc type": "PLEOMORPHIC", "calc distribution": "CLUSTERED"}
        oval = {"family": "MASS", "mass shape": "OVAL", "mass margins": "CIRCUMSCRIBED-OBSCURED"}
        dens = {"mass_density_HIGH": 0, "mass_density_LOW": 1}
        t0.update({k: torch.from_numpy(v) for k, v in A.target_tensors(
            [A.lesion_attributes(mass, dens, "MASS", 2), A.lesion_attributes(calc, dens, "CALC", 2)]).items()})
        t1.update({k: torch.from_numpy(v) for k, v in A.target_tensors(
            [A.lesion_attributes(oval, dens, "MASS", 1)]).items()})        # one lesion: density known (low)
    return imgs, [t0, t1]


def run_train(model, imgs, targets, seed=123):
    model.train()
    torch.manual_seed(seed)                     # same RPN / RoI sampling in both models
    return model(imgs, targets)


def test_l0_losses_are_identical_to_torchvision():
    base = tiny_model()
    ltn = H.install(copy.deepcopy(base))          # nothing switched on = the baseline
    imgs, targets = batch()
    lb = run_train(base, imgs, targets)
    ll = run_train(ltn, imgs, targets)
    assert set(lb) == set(ll), (set(lb), set(ll))
    for k in lb:
        assert torch.equal(lb[k], ll[k]), (k, lb[k], ll[k])


def test_l0_detections_are_identical_to_torchvision():
    base = tiny_model()
    base.roi_heads.score_thresh = 0.0             # keep many detections, so the comparison is not trivial
    ltn = H.install(copy.deepcopy(base))
    base.eval(); ltn.eval()
    imgs, _ = batch(False)
    with torch.no_grad():
        db, dl = base(imgs), ltn(imgs)
    for a, b in zip(db, dl):
        assert len(a["boxes"]) > 10
        for k in ("boxes", "scores", "labels"):
            assert torch.equal(a[k], b[k]), k


def test_concept_targets_follow_the_matched_lesion():
    imgs, targets = batch()
    matched = [torch.tensor([0, 1, 1, 0]), torch.tensor([0, 0])]
    labels = [torch.tensor([2, 3, 0, 0]), torch.tensor([1, 0])]     # region 2: matched to lesion 1 but background
    t = H.gather_attr_targets(matched, labels, targets)
    y, v = t["mass_margin"]
    assert v.tolist() == [True, False, False, False, True, False]
    assert y[0].tolist() == [0, 0, 0, 0, 1] and y[4].tolist() == [1, 1, 0, 0, 0], "sigmoid margin: both listed"
    y, v = t["calc_morph"]
    assert v.tolist() == [False, True, False, False, False, False] and y[1].tolist() == [0, 0, 1, 0, 0]
    y, v = t["mass_density"]
    assert v.tolist() == [False, False, False, False, True, False] and y[4].tolist() == [0, 1]
    assert "category" not in t


def test_class_facts_as_sat_replace_only_the_classifier_loss():
    # check run S0: the box regression and RPN losses are torchvision's, loss_classifier -> loss_class_sat
    base = tiny_model()
    s0 = H.install(copy.deepcopy(base), class_loss="sat")
    imgs, targets = batch(False)
    lb = run_train(base, imgs, targets)
    ls = run_train(s0, imgs, targets)
    assert set(ls) == (set(lb) - {"loss_classifier"}) | {"loss_class_sat"}
    for k in lb:
        if k != "loss_classifier":
            assert torch.equal(lb[k], ls[k]), k
    assert abs(s0.roi_heads.last_log["ce_class"] - float(lb["loss_classifier"].detach())) < 1e-6, "CE still logged"
    assert 0 < float(ls["loss_class_sat"]) < 1 and "sat_class" in s0.roi_heads.last_log
    assert not s0.roi_heads.use_attr_heads, "no concept heads in the check run"
    sum(ls.values()).backward()
    assert s0.roi_heads.box_predictor.cls_score.weight.grad.abs().sum() > 0


def test_ax_trains_without_concept_labels():
    model = H.install(tiny_model(), class_loss="sat", axioms="kb14", norm="region")
    model.roi_heads.ltn_p = 4.0
    imgs, targets = batch(with_attrs=False)                         # Ax: no concept labels needed
    losses = run_train(model, imgs, targets)
    assert {"loss_class_sat", "loss_clin", "loss_box_reg"} <= set(losses) and "loss_attr" not in losses
    sum(losses.values()).backward()
    for h in ("mass_shape", "mass_margin", "calc_morph", "calc_distr"):
        g = model.roi_heads.attr_heads[h].weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, h
    log = model.roi_heads.last_log
    assert any(k.startswith("sat_ax/psi") for k in log) and log["ltn_p"] == 4.0


def test_ax_c_trains_every_head():
    model = H.install(tiny_model(), class_loss="sat", attr_loss="facts", axioms="kb16")
    imgs, targets = batch()
    losses = run_train(model, imgs, targets)
    for k in ("loss_class_sat", "loss_box_reg", "loss_find", "loss_attr", "loss_clin"):
        assert k in losses and torch.isfinite(losses[k]), k
    sum(losses.values()).backward()
    for h in A.HEAD_NAMES:
        g = model.roi_heads.attr_heads[h].weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, h
    log = model.roi_heads.last_log
    assert "sat_attr/mass_density" in log and "sat_attr/mass_margin" in log and "sat_find/phi1_has_mass" in log
    assert model.roi_heads.ax_family == "pred" and model.roi_heads.class_agg == "geo", "defaults: as the paper, geo"


def test_ax_aux_adds_the_three_regularisers():
    model = H.install(tiny_model(), class_loss="sat", axioms="kb14", aux=("ent", "kl", "rec"), norm="region")
    imgs, targets = batch(with_attrs=False)
    losses = run_train(model, imgs, targets)
    assert {"loss_ent", "loss_kl", "loss_rec", "loss_clin"} <= set(losses)
    sum(losses.values()).backward()
    for h in L.AUX_HEADS["rec"]:
        assert model.roi_heads.decoders[h].weight.grad.abs().sum() > 0, h
    assert "aux_rec/mass_density" in model.roi_heads.last_log


def test_bc_uses_cross_entropy_only():
    model = H.install(tiny_model(), attr_loss="ce")
    imgs, targets = batch()
    losses = run_train(model, imgs, targets)
    assert "loss_attr_ce" in losses and "loss_classifier" in losses
    assert not {"loss_attr", "loss_clin", "loss_class_sat"} & set(losses)
    sum(losses.values()).backward()
    assert model.roi_heads.attr_heads["calc_morph"].weight.grad.abs().sum() > 0


def test_missing_concept_targets_raise():
    model = H.install(tiny_model(), class_loss="sat", attr_loss="facts")
    imgs, targets = batch(with_attrs=False)
    try:
        run_train(model, imgs, targets); raise RuntimeError("should have raised")
    except KeyError:
        pass


def test_inference_returns_concepts_per_detection():
    model = H.install(tiny_model(), class_loss="sat", attr_loss="facts", axioms="kb14")
    model.roi_heads.score_thresh = 0.0
    model.eval()
    imgs, _ = batch(False)
    with torch.no_grad():
        det = model(imgs)
    for d in det:
        n = len(d["boxes"])
        for h, k in A.HEAD_SIZES.items():
            p = d[f"attr_{h}"]
            assert p.shape == (n, k) and (p >= 0).all() and (p <= 1).all(), h
            if A.ACTIVATION[h] == "softmax":
                assert torch.allclose(p.sum(-1), torch.ones(n), atol=1e-5), h


def test_bad_values_are_refused():
    bad = [(dict(class_loss="log"), ValueError), (dict(class_agg="mean"), ValueError),
           (dict(class_bg_weight=1.0), ValueError), (dict(attr_loss="x"), ValueError),
           (dict(ax_guard="bg"), ValueError), (dict(ax_family="x"), ValueError), (dict(norm="sum"), ValueError),
           (dict(aux=("ent", "foo")), KeyError), (dict(axioms=("B1",)), KeyError), (dict(axioms=("psi15",)), ValueError),
           (dict(attr_loss="ce", axioms="kb14"), ValueError), (dict(attr_loss="ce", class_loss="sat"), ValueError),
           (dict(weights=dict(cat=1.0)), KeyError)]
    for kw, err in bad:
        try:
            H.install(tiny_model(), **kw); raise RuntimeError(f"should have raised: {kw}")
        except err:
            pass


def test_install_as_called_by_train_frcnn():
    # the exact call of build_model() in train_frcnn.py, with the defaults of the Config
    w = {"class": 1.0, "find": 1.0, "attr": 1.0, "clin": 1.0, "ent": 1.0, "kl": 1.0, "rec": 1.0}
    for agg in ("geo", "balanced"):
        imgs, targets = batch(False)
        ls = run_train(H.install(tiny_model(), class_loss="sat", class_agg=agg), imgs, targets)
        assert torch.isfinite(ls["loss_class_sat"]) and 0 < float(ls["loss_class_sat"]) < 1, agg
    for kw in (dict(), dict(attr_loss="ce"), dict(class_loss="sat"), dict(class_loss="sat", axioms="kb14"),
               dict(class_loss="sat", attr_loss="facts", axioms="kb16", aux=("ent",))):
        m = H.install(tiny_model(), class_agg="geo", class_bg_weight=0.5, ax_guard="class", ax_family="pred", ent_target=0.7,
                      weights=w, norm="region", **kw)
        assert m.roi_heads.weights == w
    imgs, targets = batch()
    half = H.install(tiny_model(), class_loss="sat", weights={"class": 0.5})
    full = H.install(tiny_model(), class_loss="sat")
    full.load_state_dict(half.state_dict())
    lh, lf = run_train(half, imgs, targets), run_train(full, imgs, targets)
    assert torch.allclose(lh["loss_class_sat"], 0.5 * lf["loss_class_sat"]), "the weight multiplies the loss"


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print(f"\n{len(tests)} tests passed")
