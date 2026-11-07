"""
What the LTN detector says about a lesion: examples and concept evaluation on the VALIDATION split.
Version 5 (SymMedNet knowledge base kb16 in the paper's notation; runs of notebook 24_ltn_v5).

Two outputs, from the best.pt of the runs (nothing is trained):

1. Concept evaluation on all validation lesions (thesis section 7.6), for every run in EVAL_RUNS:
     - every ground-truth lesion is matched to the highest-scoring detection with IoU >= 0.5 (class-agnostic);
     - on the matched detections: accuracy and macro-F1 of every head (shape, margin, density, morphology,
       distribution), class accuracy, malignancy accuracy, next to the majority-class baseline;
     - every axiom of kb16 evaluated on the PREDICTIONS of the matched detections, the same way for ALL runs (also
       C, which never trained the axioms: AxC - C is the effect of training them):
         sat          as in training (paper): over the lesions whose class label is the axiom's class (Diag guard),
                      truth of  has_mass(x, c_mass) and body => isOfClass(x, l_k)  (ApME p = 2)
         sat_label    the same with the family fixed by the lesion (has_mass left out): not vacuous when the model
                      gives the lesion a low has_mass / has_calc
         violated     share of those lesions with truth < 0.5 (family fixed)
         follows      without the class label: among the detected lesions of the axiom's family whose PREDICTED
                      concepts satisfy the body (>= 0.5), the share PREDICTED as the axiom's class (>= 0.5)
   -> concept_eval.csv, axiom_eval.csv, matched_lesions.csv

2. Example pages for EXAMPLE_RUNS (default AxC and C on G, the same images in the same order): for
   every image the mammogram with the boxes, and next to it, lesion by lesion, the annotation against the
   prediction (class, descriptors) and the axioms whose antecedent holds for that detection, with their truth.
   -> <run>_examples.pdf

Only the validation split and the official CBIS TRAINING CSVs are read (validation images are in them).
The test split is never touched.

Usage (lab machine, same Python environment as train_frcnn.py; ~5-10 min on GPU):
    python ltn_examples_and_concepts.py
Edit the CONFIG block if paths or run names differ.
"""
import os
import sys
import random
import types
import textwrap

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from sklearn.metrics import f1_score
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection._utils import Matcher
from torchvision.ops import boxes as box_ops

# ==================================================================================================
# CONFIG
# ==================================================================================================
CODE_DIR = r"C:\Users\grains7lab2\Desktop\Tesi_Karimov\code\ltn_v5"     # ltn_attributes / ltn_logic / ltn_heads v5
DATA_ROOT = r"F:\MammoLTN-full-images"
PNG_ROOT = os.path.join(DATA_ROOT, "CBIS-DDSM-PNG16-CROPPED-MASKS")
IMAGES_DIR = os.path.join(PNG_ROOT, "images")
MASKS_DIR = os.path.join(PNG_ROOT, "lesion_masks")
MANIFEST_CSV = os.path.join(PNG_ROOT, "manifest.csv")
SPLITS_XLSX = os.path.join(DATA_ROOT, "CBIS_08_11_2024", "cbis_ddsm_image_level_combined.xlsx")
CSV_DIR = os.path.join(DATA_ROOT, "CBIS_08_11_2024")
CSV_FILES = ("mass_case_description_train_set.csv", "calc_case_description_train_set.csv")

RUNS = r"F:\runs"
EVAL_RUNS = {            # label -> run folder (missing folders are skipped); runs without concept heads are skipped
    f"{s} {b}": f"v5_{b}_{s}_geo_e50" for b in ("G", "GV") for s in ("C", "AxC", "Ax")
}
EXAMPLE_RUNS = {"AxC on G (facts + axioms)": "v5_G_AxC_geo_e50", "C on G (facts, no axioms)": "v5_G_C_geo_e50"}
OUT_DIR = os.path.join(RUNS, "ltn_examples_v5")

N_PER_CLASS = 4          # example images per ground-truth class (16 images)
SAMPLE_SEED = 0          # fixes the example images; the same for every model
IOU_HIT = 0.50           # a detection matches a lesion if IoU >= this (as froc_iou in train_frcnn.py)
SHOW_THR = 0.30          # other detections drawn (dashed) if score >= this
MAX_OTHER = 3            # at most this many other detections drawn per image
AXIOM_SHOW = 0.30        # list an axiom on the page if its antecedent is >= this
P_APME = 2.0             # ApME exponent, as in training

sys.path.insert(0, CODE_DIR)
import ltn_attributes as A      # noqa: E402
import ltn_logic as L           # noqa: E402
import ltn_heads as H           # noqa: E402

CLASS_NAMES = {0: "background", 1: "mass benign", 2: "mass malignant", 3: "calc benign", 4: "calc malignant"}
SHORT = {1: "MB", 2: "MM", 3: "CB", 4: "CM"}
COLORS = {1: "#2c6fbb", 2: "#c0392b", 3: "#17a2a2", 4: "#e08a1e"}
LABEL = {("MASS", "BENIGN"): 1, ("MASS", "MALIGNANT"): 2, ("CALC", "BENIGN"): 3, ("CALC", "MALIGNANT"): 4}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AX_TEXT = {"psi3": "mass circumscribed/obscured -> benign", "psi4": "mass microlobulated -> malignant",
           "psi5": "mass indistinct -> malignant", "psi6": "mass irregular -> malignant",
           "psi7": "amorphous, not regional/diffuse -> malignant", "psi8": "pleomorphic, not regional/diffuse -> malignant",
           "psi9": "mass high density -> malignant", "psi10": "mass spiculated -> malignant",
           "psi11": "fine linear + segmental -> malignant", "psi12": "fine linear + linear -> malignant",
           "psi13": "coarse het., not regional/diffuse -> malignant", "psi14": "fine linear + clustered -> benign",
           "psi16": "regional -> benign", "N1": "typ. benign, no suspicious -> benign",
           # (N1-N3: proposals not in kb16, kept only for older runs)
           "N2": "fine linear -> malignant", "N3": "diffuse -> benign"}
AXIOMS_EVAL = list(L.AXIOM_SETS["kb16"])      # psi3-psi14, psi16 (call of 6 Nov)
CLASS_OF = {k: set(v) for k, v in L.CLASS_SETS.items()}


# ==================================================================================================
# Model
# ==================================================================================================
def build_model(cfg: dict) -> torch.nn.Module:
    """The architecture of build_model() in train_frcnn.py, from the config saved in best.pt (no download)."""
    norm = cfg.get("normalization", "imagenet")
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if norm == "imagenet" else ([0.0] * 3, [1.0] * 3)
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                    min_size=cfg["image_size"], max_size=cfg["image_size"],
                                    image_mean=mean, image_std=std)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, len(CLASS_NAMES))
    sizes = tuple(tuple(float(s) for s in lvl) for lvl in cfg["anchor_sizes"])
    ratios = tuple(float(r) for r in cfg["anchor_ratios"])
    model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(ratios,) * len(sizes))
    n = model.rpn.anchor_generator.num_anchors_per_location()[0]
    if n != model.rpn.head.cls_logits.out_channels:
        model.rpn.head = RPNHead(model.backbone.out_channels, n)
    model.rpn.proposal_matcher = Matcher(cfg["rpn_fg_iou"], cfg["rpn_bg_iou"], allow_low_quality_matches=True)
    if cfg.get("ltn"):
        H.install(model, class_loss=cfg.get("ltn_class_loss", "ce"), class_agg=cfg.get("ltn_class_agg", "geo"),
                  class_bg_weight=cfg.get("ltn_class_bg_w", 0.5), attr_loss=cfg.get("ltn_attr_loss", "none"),
                  axioms=cfg.get("ltn_axioms") or (), ax_guard=cfg.get("ltn_ax_guard", "class"),
                  ax_family=cfg.get("ltn_ax_family", "pred"), aux=tuple(cfg.get("ltn_aux") or ()),
                  norm=cfg.get("ltn_norm", "mean"))       # the losses are not used here: only the modules matter
        model.roi_heads._postprocess = types.MethodType(_postprocess_with_class_probs, model.roi_heads)
    return model


def _postprocess_with_class_probs(self, class_logits, box_regression, attr_logits, proposals, image_shapes):
    """LTNRoIHeads._postprocess, line by line, plus the full class softmax of the region behind every detection
    (det["class_probs"], [n, 5]): the axioms need Mass, Calc, Benign, Malignant, not only the winning score."""
    device = class_logits.device
    num_classes = class_logits.shape[-1]
    boxes_per_image = [b.shape[0] for b in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)
    pred_scores = F.softmax(class_logits, -1)
    attr_probs = L.head_probs({h: z.float() for h, z in attr_logits.items()}, A.ACTIVATION)
    out, start = [], 0
    for boxes, scores, image_shape, n in zip(pred_boxes.split(boxes_per_image, 0),
                                             pred_scores.split(boxes_per_image, 0), image_shapes, boxes_per_image):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        labels = torch.arange(num_classes, device=device).view(1, -1).expand_as(scores)
        region = torch.arange(start, start + n, device=device).view(-1, 1).expand_as(scores)
        boxes, scores, labels, region = boxes[:, 1:], scores[:, 1:], labels[:, 1:], region[:, 1:]
        boxes, scores = boxes.reshape(-1, 4), scores.reshape(-1)
        labels, region = labels.reshape(-1), region.reshape(-1)
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels, region = boxes[inds], scores[inds], labels[inds], region[inds]
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels, region = boxes[keep], scores[keep], labels[keep], region[keep]
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[: self.detections_per_img]
        det = {"boxes": boxes[keep], "labels": labels[keep], "scores": scores[keep],
               "class_probs": pred_scores[region[keep]]}
        for h, p in attr_probs.items():
            det[f"attr_{h}"] = p[region[keep]]
        out.append(det)
        start += n
    return out


# best.pt of a run -> the model (config read from the checkpoint), best epoch, its mAP@50
def load_model(run: str):
    ck = torch.load(os.path.join(RUNS, run, "best.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if not cfg.get("ltn") or (cfg.get("ltn_attr_loss", "none") == "none" and not cfg.get("ltn_axioms")
                              and not cfg.get("ltn_aux")):
        raise ValueError(f"{run} has no concept heads (not an LTN run with concept heads)")
    model = build_model(cfg)
    model.load_state_dict(ck["model"])            # strict: fails loudly if the code version does not match the run
    model.to(DEVICE).eval()
    m = ck.get("metrics", {}) or {}
    return model, cfg, ck.get("epoch"), m.get("map_50", m.get("val/map_50"))


# ==================================================================================================
# Data: validation lesions with their boxes and BI-RADS facts
# ==================================================================================================
# validation images (no mixed pathology, as train_frcnn.py) with their lesions: box from the mask and the
# BI-RADS facts (the same encoding as the facts file)
def validation_records():
    df = pd.read_excel(SPLITS_XLSX)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["split"] == "Valid"].copy()                      # validation only; test is never read
    mixed = df["source_patch_labels"].astype(str).map(lambda s: "BENIGN" in s and "MALIGNANT" in s)
    df = df[~mixed]
    df["image_id"] = df["image_filename"].str.replace("_crop.png", "", regex=False)
    index, _ = A.load_cbis_csvs([os.path.join(CSV_DIR, f) for f in CSV_FILES], set(df["image_id"]))
    man = pd.read_csv(MANIFEST_CSV)
    man.columns = [c.strip() for c in man.columns]
    man = man[man["status"].astype(str).str.lower() == "ok"]
    by_img = {k: v for k, v in man.groupby("image_id")}
    recs = []
    for _, row in df.iterrows():
        iid = row["image_id"]
        fam = "MASS" if iid.startswith("Mass") else "CALC"
        path = str(row["label_upd"]).strip().upper()
        ip = os.path.join(IMAGES_DIR, f"{iid}_crop.png")
        if path not in ("BENIGN", "MALIGNANT") or iid not in by_img or not os.path.exists(ip):
            continue
        mrows = by_img[iid]
        n_les = int(mrows["lesion_id"].notna().sum())
        lesions = []
        for lid in mrows["lesion_id"].dropna():
            m = cv2.imread(os.path.join(MASKS_DIR, f"{iid}_{int(lid)}_mask.png"), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim == 3:
                m = m[..., 0]
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            facts = A.lesion_attributes(index.get((iid, int(lid))), row, fam, n_les) or {}
            lesions.append(dict(lesion_id=int(lid), box=[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], facts=facts))
        if lesions:
            recs.append(dict(image_id=iid, path=ip, family=fam, label=LABEL[(fam, path)], lesions=lesions))
    return recs


# grey image resized to the training size, in [0, 1]; returns it and the scale
def load_image(path: str, size: int):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = img[..., 0]
    full = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    return cv2.resize(img.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) / full, s


# IoU matrix between two lists of boxes [x0, y0, x1, y1]
def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0, y0 = np.maximum(a[:, None, :2], b[None, :, :2]).transpose(2, 0, 1)
    x1, y1 = np.minimum(a[:, None, 2:], b[None, :, 2:]).transpose(2, 0, 1)
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


@torch.no_grad()
# one image -> detections (boxes, scores, class_probs, attr_<head>)
def predict(model, img):
    x = torch.from_numpy(img)[None].repeat(3, 1, 1).to(DEVICE)
    o = model([x])[0]
    return {k: v.cpu() for k, v in o.items()}


def match(det, gt_boxes):
    """For every GT lesion: index of the highest-scoring detection with IoU >= IOU_HIT, or -1."""
    ov = iou(det["boxes"].numpy(), np.asarray(gt_boxes, float))
    out = []
    for j in range(ov.shape[1]):
        cand = np.where(ov[:, j] >= IOU_HIT)[0]
        out.append(int(cand[np.argmax(det["scores"].numpy()[cand])]) if len(cand) else -1)
    return out


def detection_probs(det, k):
    """Class predicates and head probabilities of detection k, as [1, ...] tensors for ltn_logic."""
    cls = L.class_predicates(det["class_probs"][k:k + 1].clamp_min(1e-12).log())
    probs = {h: det[f"attr_{h}"][k:k + 1] for h in A.HEAD_NAMES}
    return cls, probs


def antecedent(name, cls, probs):
    """The body of the axiom on the predicted CONCEPTS only (has_mass / has_calc left out: the family is the lesion's)."""
    ax = L.AXIOMS[name]
    parts = [torch.ones_like(cls["c_mass"])]
    for pred, c in ax["body"]:
        if pred not in ("has_mass", "has_calc"):
            parts.append(L.literal(pred, c, cls, probs, A.VALUE_INDEX, A.ACTIVATION))
    for pred, c in ax["nots"]:
        parts.append(L.Not(L.literal(pred, c, cls, probs, A.VALUE_INDEX, A.ACTIVATION)))
    return L.And(*parts)


# ==================================================================================================
# 1. Concept evaluation
# ==================================================================================================
# one row per validation lesion: matched detection (if any), its class / concept probabilities, and for every
# axiom: truth as in training (truth_), with the family fixed (truthL_), antecedent (ante_), consequent (cons_)
def evaluate(model, recs, size):
    rows = []
    for r in recs:
        img, s = load_image(r["path"], size)
        det = predict(model, img)
        gt = [np.array(le["box"], float) * s for le in r["lesions"]]
        for le, k in zip(r["lesions"], match(det, gt)):
            row = dict(image_id=r["image_id"], lesion_id=le["lesion_id"], family=r["family"], gt_class=r["label"],
                       detected=k >= 0)
            for h in A.HEAD_NAMES:
                row[f"gt_{h}"] = le["facts"].get(h)
            if k >= 0:
                cls, probs = detection_probs(det, k)
                cp = det["class_probs"][k].numpy()
                row.update(score=float(det["scores"][k]), pred_class=int(np.argmax(cp[1:]) + 1),
                           p_malignant=float(cp[2] + cp[4]))
                for h in A.HEAD_NAMES:
                    row[f"p_{h}"] = probs[h][0].numpy().round(4).tolist()
                for a in AXIOMS_EVAL:
                    row[f"truth_{a}"] = float(L.axiom_truths(a, cls, probs, A.VALUE_INDEX, A.ACTIVATION, "pred")[0])
                    row[f"truthL_{a}"] = float(L.axiom_truths(a, cls, probs, A.VALUE_INDEX, A.ACTIVATION, "label")[0])
                    row[f"ante_{a}"] = float(antecedent(a, cls, probs)[0])
                    row[f"cons_{a}"] = float(cls[L.AXIOMS[a]["head"]][0])
            rows.append(row)
    return pd.DataFrame(rows)


def concept_metrics(df):
    """Accuracy / macro-F1 of every head on the matched detections that have a fact for it."""
    out = {}
    m = df[df["detected"]]
    out["lesions"] = len(df)
    out["detected (IoU >= 0.5)"] = float(df["detected"].mean())
    if not len(m):
        return out
    out["class accuracy"] = float((m["pred_class"] == m["gt_class"]).mean())
    gt_mal = m["gt_class"].isin([2, 4])
    out["malignancy accuracy"] = float(((m["p_malignant"] >= 0.5) == gt_mal).mean())
    for h in A.HEAD_NAMES:
        sub = m[m[f"gt_{h}"].notna()]
        if not len(sub):
            continue
        y = np.array(sub[f"gt_{h}"].tolist(), float)
        p = np.array(sub[f"p_{h}"].tolist(), float)
        if A.ACTIVATION[h] == "softmax":
            pred = p.argmax(1)
            ok = y[np.arange(len(y)), pred] > 0
            true = np.where(ok, pred, y.argmax(1))              # a tie (disjunction) counts as right if either value
            maj = np.bincount(y.argmax(1), minlength=y.shape[1]).max() / len(y)
            out[f"{h} accuracy"] = float(ok.mean())
            out[f"{h} macro-F1"] = float(f1_score(true, pred, labels=list(range(y.shape[1])), average="macro",
                                                  zero_division=0))
            out[f"{h} majority baseline"] = float(maj)
        else:
            pred = (p >= 0.5).astype(int)
            cols = np.where(y.sum(0) > 0)[0]                     # values that occur (coarse heterogeneous does not)
            out[f"{h} accuracy (per value)"] = float((pred[:, cols] == y[:, cols]).mean())
            out[f"{h} macro-F1"] = float(f1_score(y[:, cols], pred[:, cols], average="macro", zero_division=0))
            out[f"{h} majority baseline"] = float(np.mean([max(y[:, c].mean(), 1 - y[:, c].mean()) for c in cols]))
    return out


def axiom_metrics(df):
    """Every axiom on the predictions of the matched detections.
       sat, violated: over the lesions of the axiom's family whose class label is the consequent (as in training);
       follows: over the lesions of the axiom's family whose PREDICTED antecedent holds (>= 0.5), the share whose
                PREDICTED class is the consequent (>= 0.5); no class label used."""
    m = df[df["detected"]]
    rows = []
    if not len(m):
        return rows
    for a in AXIOMS_EVAL:
        ax = L.AXIOMS[a]
        fam = m["family"] == L.axiom_family(a)
        cls_ok = m["gt_class"].isin(CLASS_OF[ax["guard"]])
        dom_paper = cls_ok                                    # Diag(x, l_k): every lesion of the class (both families)
        dom_fam = fam & cls_ok
        t = torch.tensor(m.loc[dom_paper, f"truth_{a}"].values, dtype=torch.float32)
        tl = torch.tensor(m.loc[dom_fam, f"truthL_{a}"].values, dtype=torch.float32)
        sat = L.apme(t, P_APME) if len(t) else None
        satl = L.apme(tl, P_APME) if len(tl) else None
        on = fam & (m[f"ante_{a}"] >= 0.5)
        rows.append(dict(axiom=a, rule=AX_TEXT[a], text=L.axiom_text(a), domain=int(dom_paper.sum()),
                         sat=float(sat) if sat is not None else np.nan,
                         domain_family=int(dom_fam.sum()), sat_label=float(satl) if satl is not None else np.nan,
                         violated=float((tl < 0.5).float().mean()) if len(tl) else np.nan,
                         antecedent_true=int(on.sum()),
                         follows=float((m.loc[on, f"cons_{a}"] >= 0.5).mean()) if on.any() else np.nan,
                         label_agrees=float(m.loc[on, "gt_class"].isin(CLASS_OF[ax["guard"]]).mean())
                         if on.any() else np.nan))
    return rows


# ==================================================================================================
# 2. Example pages
# ==================================================================================================
# text of the annotated value(s) for the example pages
def fmt_fact(h, v):
    if v is None:
        return "—"
    names = A.HEADS[h][2]
    on = [names[i] for i, x in enumerate(v) if x > 0]
    return " or ".join(n.replace("_", " ") for n in on) if A.ACTIVATION[h] == "softmax" else " + ".join(
        n.replace("_", " ") for n in on)


# text of the predicted value(s) for the example pages
def fmt_pred(h, p):
    names = A.HEADS[h][2]
    if A.ACTIVATION[h] == "sigmoid":
        on = [f"{names[i].replace('_', ' ')} {x:.2f}" for i, x in enumerate(p) if x >= 0.5]
        return " + ".join(on) if on else f"none >= 0.5 (top: {names[int(np.argmax(p))].replace('_', ' ')} {max(p):.2f})"
    i = int(np.argmax(p))
    n = names[i].replace("_", " ")
    return f"{n} {p[i]:.2f}"


# lines next to the image: annotation, prediction, rules whose antecedent holds
def lesion_text(r, le, det, k):
    fam = r["family"]
    heads = ["mass_shape", "mass_margin", "mass_density"] if fam == "MASS" else ["calc_morph", "calc_distr"]
    lab = {"mass_shape": "shape", "mass_margin": "margin", "mass_density": "density", "calc_morph": "morphology",
           "calc_distr": "distribution"}
    f = le["facts"]
    lines = [f"GT   {CLASS_NAMES[r['label']]}"]
    lines.append("     " + ";  ".join(f"{lab[h]}: {fmt_fact(h, f.get(h))}" for h in heads))
    if k < 0:
        lines.append("PRED not detected (no box with IoU >= 0.5)")
        return lines
    cls, probs = detection_probs(det, k)
    cp = det["class_probs"][k].numpy()
    c = int(np.argmax(cp[1:]) + 1)
    lines.append(f"PRED {CLASS_NAMES[c]} {cp[c]:.2f} (malignant {cp[2] + cp[4]:.2f}, box score "
                 f"{float(det['scores'][k]):.2f})")
    lines.append("     " + ";  ".join(f"{lab[h]}: {fmt_pred(h, probs[h][0].numpy())}" for h in heads))
    act = []
    for a in AXIOMS_EVAL:
        if L.axiom_family(a) != fam:
            continue
        ante = float(antecedent(a, cls, probs)[0])
        if ante >= AXIOM_SHOW:
            t = float(L.axiom_truths(a, cls, probs, A.VALUE_INDEX, A.ACTIVATION, "label")[0])
            act.append(f"{a} ({AX_TEXT[a]}): {t:.2f} {'ok' if t >= 0.5 else 'VIOLATED'}")
    lines.append("AXIOMS that apply (antecedent >= %.1f):" % AXIOM_SHOW)
    lines += ["     " + x for x in act] if act else ["     none"]
    return lines


# first page: legend + summary of the concept evaluation; then two images per page
def example_pdf(name, run, model, cfg, epoch, m50, chosen, summary, pdf_path):
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.92, name, fontsize=16, weight="bold")
        fig.text(0.07, 0.885, f"run {run}   best epoch {epoch}" + (f"   validation mAP@50 {m50:.3f}" if m50 else ""),
                 fontsize=10)
        txt = ["Every page: two validation images. Yellow = annotated lesion (number = lesion id).",
               "Solid box = the detection matched to the lesion (highest score with IoU >= 0.5);",
               f"dashed = other detections with score >= {SHOW_THR}. Next to the image, per lesion:",
               "GT = annotation (class = biopsy pathology of the image; descriptors from the CBIS CSV),",
               "PRED = what the model says for the matched box; AXIOMS = rules of the lesion's family whose",
               "antecedent holds, with the truth of the rule on the model's own predictions (ok >= 0.5).", "",
               "Concept evaluation on ALL validation lesions (matched detections):"]
        txt += [f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}" for k, v in summary.items()]
        fig.text(0.07, 0.86, "\n".join(txt), fontsize=8.5, va="top", family="monospace")
        pdf.savefig(fig); plt.close(fig)
        for p0 in range(0, len(chosen), 2):
            fig = plt.figure(figsize=(8.27, 11.69))
            for slot, r in enumerate(chosen[p0:p0 + 2]):
                img, s = load_image(r["path"], cfg["image_size"])
                det = predict(model, img)
                gt = [np.array(le["box"], float) * s for le in r["lesions"]]
                ks = match(det, gt)
                top = 0.53 - slot * 0.49
                ax = fig.add_axes([0.03, top, 0.36, 0.40])
                ax.imshow(img, cmap="gray", vmin=0, vmax=max(float(img.max()), 1e-6)); ax.axis("off")
                for le, b in zip(r["lesions"], gt):
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#f1c40f", lw=1.6))
                    ax.text(b[2] + 2, b[1], str(le["lesion_id"]), color="#f1c40f", fontsize=7, weight="bold")
                bx, sc = det["boxes"].numpy(), det["scores"].numpy()
                matched = set(k for k in ks if k >= 0)
                others = [k for k in range(len(sc)) if k not in matched and sc[k] >= SHOW_THR][:MAX_OTHER]
                for k in list(matched) + others:
                    b = bx[k]; c = int(det["labels"][k])
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=COLORS[c], lw=1.3,
                                           ls="-" if k in matched else "--"))
                    ax.text(b[0], b[1] - 2, f"{SHORT[c]} {sc[k]:.2f}", color="white", fontsize=6,
                            bbox=dict(facecolor=COLORS[c], edgecolor="none", pad=0.8))
                ax.set_title(f"{r['image_id']}  ({CLASS_NAMES[r['label']]})", fontsize=7.5, pad=3)
                lines = []
                for le, k in zip(r["lesions"][:3], ks[:3]):
                    lines.append(f"Lesion {le['lesion_id']}")
                    for t in lesion_text(r, le, det, k):
                        lines += textwrap.wrap(t, 74, subsequent_indent="       ")
                    lines.append("")
                if len(r["lesions"]) > 3:
                    lines.append(f"(+ {len(r['lesions']) - 3} more lesions not listed)")
                fig.text(0.41, top + 0.40, "\n".join(lines), fontsize=7, va="top", family="monospace")
            fig.suptitle(name, fontsize=9, weight="bold", y=0.995)
            pdf.savefig(fig); plt.close(fig)


# ==================================================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = validation_records()
    print(f"{len(recs)} validation images, {sum(len(r['lesions']) for r in recs)} lesions")
    rng = random.Random(SAMPLE_SEED)
    chosen = []
    for lab in (1, 2, 3, 4):
        pool = sorted([r for r in recs if r["label"] == lab and any(le["facts"] for le in r["lesions"])],
                      key=lambda r: r["image_id"])
        chosen += rng.sample(pool, min(N_PER_CLASS, len(pool)))
    pd.DataFrame([dict(image_id=r["image_id"], gt=CLASS_NAMES[r["label"]], n_lesions=len(r["lesions"]))
                  for r in chosen]).to_csv(os.path.join(OUT_DIR, "chosen_images.csv"), index=False)

    concept_rows, axiom_rows, summaries = [], [], {}
    runs = dict(EVAL_RUNS)
    for name, run in EXAMPLE_RUNS.items():
        if run not in runs.values():
            runs[name] = run
    for label, run in runs.items():
        if not os.path.exists(os.path.join(RUNS, run, "best.pt")):
            print(f"skip {label}: {run}/best.pt not found"); continue
        try:
            model, cfg, epoch, m50 = load_model(run)
        except ValueError as e:                     # e.g. S0 or G: no concept heads
            print(f"skip {label}: {e}"); continue
        df = evaluate(model, recs, cfg["image_size"])
        df.assign(run=label).to_csv(os.path.join(OUT_DIR, f"matched_lesions_{run}.csv"), index=False)
        cm = concept_metrics(df)
        summaries[run] = cm
        concept_rows.append(dict(run=label, folder=run, best_epoch=epoch, mAP50=m50, **cm))
        axiom_rows += [dict(run=label, folder=run, **x) for x in axiom_metrics(df)]
        print(f"{label}: detected {cm['detected (IoU >= 0.5)']:.3f}, class acc {cm.get('class accuracy', np.nan):.3f}, "
              f"malignancy acc {cm.get('malignancy accuracy', np.nan):.3f}")
        for name, ex_run in EXAMPLE_RUNS.items():
            if ex_run == run:
                pdf_path = os.path.join(OUT_DIR, f"{run}_examples.pdf")
                example_pdf(name, run, model, cfg, epoch, m50, chosen, cm, pdf_path)
                print("   examples ->", pdf_path)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(concept_rows).to_csv(os.path.join(OUT_DIR, "concept_eval.csv"), index=False)
    ax = pd.DataFrame(axiom_rows)
    ax.to_csv(os.path.join(OUT_DIR, "axiom_eval.csv"), index=False)
    if len(ax):
        pd.set_option("display.width", 220)
        print("\nSat of every axiom on the matched validation detections, as in training (ApME p=2):")
        print(ax.pivot(index="axiom", columns="run", values="sat").reindex(AXIOMS_EVAL).round(3).to_string())
        print("\nSat with the family fixed by the lesion (has_mass / has_calc left out):")
        print(ax.pivot(index="axiom", columns="run", values="sat_label").reindex(AXIOMS_EVAL).round(3).to_string())
        print("\nShare of the axiom's lesions (its class and family) that VIOLATE it (truth < 0.5):")
        print(ax.pivot(index="axiom", columns="run", values="violated").reindex(AXIOMS_EVAL).round(3).to_string())
        print("\nWhere the predicted antecedent holds: share predicted as the consequent (follows the rule):")
        print(ax.pivot(index="axiom", columns="run", values="follows").reindex(AXIOMS_EVAL).round(3).to_string())
    print("\ndone ->", OUT_DIR, "  send concept_eval.csv, axiom_eval.csv and the *_examples.pdf files")


if __name__ == "__main__":
    main()
