"""
LTN RoI heads for torchvision's Faster R-CNN (thesis Chapter 6). Version 5 (call of 6 Nov 2026): knowledge base of
SymMedNet (appendix B.9) in the paper's notation (ltn_logic.py), facts read from the facts file (ltn_attributes.py).

LTNRoIHeads is torchvision's RoIHeads with these additions; everything else (sampling, RoIAlign, box head, class head,
box regressor, box loss, NMS) is torchvision's own code path, so with nothing switched on the model IS the baseline.

  1. five concept heads on the region embedding z (one Linear 1024 -> K each): shape, density (2 outputs) and
     distribution with a softmax; margin and morphology with a sigmoid (ltn_attributes.ACTIVATION).
  2. in training, the losses of ltn_logic.kb_losses (every one is 1 - Sat):
       class_loss="sat"  -> loss_class_sat        phi_class: forall Diag(x, l_k). isOfClass(x, l_k)
       attr_loss="facts" -> loss_find, loss_attr  phi1-phi2 (has_mass, has_calc) and phi3-phi21 (concepts)
       axioms=(...)      -> loss_clin             psi, each forall Diag(x, l_k) over its class
       aux=(...)         -> loss_ent, loss_kl, loss_rec   SymMedNet "+Aux" regularisers (not in the plan of 6 Nov)
     class_loss="ce" (the baseline) keeps torchvision's cross-entropy; attr_loss="ce" = BC (R50+C, no logic).
     Plan of 6 Nov: S0 = class facts only (the baseline converted to LTN), C = + facts, AxC = + facts + axioms,
     Ax = axioms only. The box regression loss and the RPN losses are always torchvision's.
  3. at inference, every detection also carries the probabilities of its heads (attr_<head>, [n, K]).

Individuals = the regions sampled by torchvision for the box head (512 per image, ground-truth boxes included).
Foreground regions (label > 0, IoU >= 0.5) inherit the concepts of the lesion they are matched to.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops

import ltn_attributes as A
import ltn_logic as L

VERSION = 5
WEIGHT_KEYS = ("class", "find", "attr", "clin", "ent", "kl", "rec")


def gather_attr_targets(matched_idxs: List[torch.Tensor], labels: List[torch.Tensor],
                        targets: List[Dict[str, torch.Tensor]]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Concept targets of the sampled regions, concatenated over the images in the order of the regions.
    A region gets the concepts of the lesion it is matched to, and only if it is foreground (label > 0) and that
    lesion has a value for the head (attr_<head>_valid). Background regions are never valid."""
    out = {}
    for head, k in A.HEAD_SIZES.items():
        ys, vs = [], []
        for midx, lab, t in zip(matched_idxs, labels, targets):
            y_img, v_img = t[f"attr_{head}"], t[f"attr_{head}_valid"]
            if y_img.shape[0] == 0:                           # image without lesions: every region is background
                ys.append(torch.zeros((len(midx), k), device=midx.device))
                vs.append(torch.zeros(len(midx), dtype=torch.bool, device=midx.device))
                continue
            if y_img.shape[1] != k:
                raise ValueError(f"head {head}: targets with {y_img.shape[1]} values, the head has {k}")
            # midx = index of the ground-truth box matched to every region (torchvision's matcher). Background regions
            # also get an index, so a target is valid only if the region is foreground and the lesion has that fact
            ys.append(y_img[midx].float())
            vs.append(v_img[midx].bool() & (lab > 0))
        out[head] = (torch.cat(ys), torch.cat(vs))
    return out


class LTNRoIHeads(RoIHeads):
    def __init__(self, base: RoIHeads, class_loss: str = "ce", class_agg: str = "geo", class_bg_weight: float = 0.5,
                 attr_loss: str = "none",
                 axioms: Union[str, Sequence[str]] = (), ax_guard: str = "class", ax_family: str = "pred",
                 aux: Sequence[str] = (),
                 ent_target: float = 0.7, weights: Optional[Dict[str, float]] = None,
                 norm: str = "mean"):
        # torchvision's RoIHeads rebuilt with the SAME modules (RoIAlign, box head, box predictor) and settings,
        # so the pretrained / baseline weights are kept
        super().__init__(
            base.box_roi_pool, base.box_head, base.box_predictor,
            base.proposal_matcher.high_threshold, base.proposal_matcher.low_threshold,
            base.fg_bg_sampler.batch_size_per_image, base.fg_bg_sampler.positive_fraction,
            base.box_coder.weights, base.score_thresh, base.nms_thresh, base.detections_per_img)
        # every option is checked here: a wrong config fails at start, not after hours of training
        if base.has_mask() or base.has_keypoint():
            raise ValueError("LTNRoIHeads supports boxes only")
        if class_loss not in ("ce", "sat"):
            raise ValueError(f"class_loss {class_loss!r}: use 'ce' or 'sat'")
        if class_agg not in L.CLASS_AGGS:
            raise ValueError(f"class_agg {class_agg!r}: use one of {L.CLASS_AGGS}")
        if not 0.0 < class_bg_weight < 1.0:
            raise ValueError(f"class_bg_weight {class_bg_weight}: must be in (0, 1)")
        if attr_loss not in ("none", "ce", "facts"):
            raise ValueError(f"attr_loss {attr_loss!r}: use 'none', 'ce' or 'facts'")
        if ax_guard not in ("class", "fg", "all"):
            raise ValueError(f"ax_guard {ax_guard!r}: use 'class', 'fg' or 'all'")
        if ax_family not in ("pred", "label"):
            raise ValueError(f"ax_family {ax_family!r}: use 'pred' (as the paper) or 'label'")
        if norm not in ("mean", "region"):
            raise ValueError(f"norm {norm!r}: use 'mean' or 'region'")
        aux = tuple(aux or ())
        bad_aux = [a for a in aux if a not in L.AUX_HEADS]
        if bad_aux or len(set(aux)) != len(aux):
            raise KeyError(f"aux {aux}: use a subset of {list(L.AUX_HEADS)}")
        self.class_loss, self.class_agg, self.attr_loss = class_loss, class_agg, attr_loss
        self.class_bg_weight = float(class_bg_weight)
        self.axioms = L.resolve_axioms(axioms)
        self.ax_guard, self.ax_family = ax_guard, ax_family
        self.aux, self.ent_target, self.norm = aux, float(ent_target), norm
        if attr_loss == "ce" and (self.axioms or class_loss == "sat"):
            raise ValueError("attr_loss='ce' is the R50+C baseline (BC): no logic with it; use attr_loss='facts'")
        w = {k: 1.0 for k in WEIGHT_KEYS}
        unknown_w = set(weights or {}) - set(WEIGHT_KEYS)
        if unknown_w:
            raise KeyError(f"unknown loss weights {sorted(unknown_w)}: use {list(WEIGHT_KEYS)}")
        w.update(weights or {})
        self.weights = w
        # concept heads exist when something trains or uses them (facts, CE, axioms, aux)
        self.use_attr_heads = attr_loss != "none" or bool(self.axioms) or bool(aux)
        in_features = base.box_predictor.cls_score.in_features
        if self.use_attr_heads:
            # one Linear per concept head on the 1024-d region embedding z, the same z the class head reads
            self.attr_heads = nn.ModuleDict({h: nn.Linear(in_features, k) for h, k in A.HEAD_SIZES.items()})
        if "rec" in aux:
            self.decoders = nn.ModuleDict({h: nn.Linear(A.HEAD_SIZES[h], in_features) for h in L.AUX_HEADS["rec"]})
        missing_h = sorted({h for a in self.axioms for h in L.axiom_heads(a)} - set(A.HEAD_SIZES))
        if missing_h:
            raise KeyError(f"the axioms need heads {missing_h} that ltn_attributes does not define")
        self.ltn_p = 2.0                      # ApME exponent, set every epoch by the training loop
        self.last_log: Dict[str, float] = {}  # satisfaction / CE values of the last forward, for logging

    @property
    def ltn_active(self) -> bool:
        return self.use_attr_heads or self.class_loss == "sat"

    def describe(self) -> str:
        agg = self.class_agg + (f", bg {self.class_bg_weight}" if self.class_agg == "balanced" else "")
        return (f"class_loss={self.class_loss} (agg {agg})  attr_loss={self.attr_loss}  "
                f"axioms={list(self.axioms) or '-'}  ax_guard={self.ax_guard}  ax_family={self.ax_family}  "
                f"aux={list(self.aux) or '-'}  "
                f"norm={self.norm}  heads={'yes' if self.use_attr_heads else 'no'}  "
                f"weights={self.weights}")

    # ---------------------------------------------------------------------------------------------- forward
    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                if t["boxes"].dtype not in (torch.float, torch.double, torch.half):
                    raise TypeError(f"target boxes must be float, got {t['boxes'].dtype}")
                if t["labels"].dtype != torch.int64:
                    raise TypeError(f"target labels must be int64, got {t['labels'].dtype}")

        if self.training:
            # torchvision: 512 regions per image (at most 25% foreground), each with its class label and matched box
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # z = region embedding [R, 1024]: input of the class, box and concept heads
        z = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(z)
        attr_logits = {h: m(z) for h, m in self.attr_heads.items()} if self.use_attr_heads else {}

        result: List[Dict[str, torch.Tensor]] = []
        losses: Dict[str, torch.Tensor] = {}
        self.last_log = {}
        if self.training:
            # torchvision's own losses; the box regression loss is always kept
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_box_reg": loss_box_reg}
            # baseline: cross-entropy; LTN ('sat'): the CE is only logged and loss_class_sat replaces it
            if self.class_loss == "ce":
                losses["loss_classifier"] = loss_classifier
            else:
                self.last_log["ce_class"] = float(loss_classifier.detach())   # same number as G's loss_classifier
            if self.ltn_active:
                losses.update(self._ltn_losses(class_logits, attr_logits, z, proposals, matched_idxs, labels, targets))
        else:
            # inference: torchvision's postprocessing + concept probabilities
            for det in self._postprocess(class_logits, box_regression, attr_logits, proposals, image_shapes):
                result.append(det)
        return result, losses

    def _ltn_losses(self, class_logits, attr_logits, z, proposals, matched_idxs, labels, targets):
        # the logic runs in float32 even under mixed precision (powers and products of probabilities)
        class_logits = class_logits.float()
        attr_logits = {h: v.float() for h, v in attr_logits.items()}
        # the regions of all images of the batch: the quantifiers run over all of them
        labels_cat = torch.cat(labels)
        out = {}
        attr_t = None
        if self.attr_loss != "none":
            if f"attr_{A.HEAD_NAMES[0]}" not in targets[0]:
                raise KeyError("concept targets missing: run with --ltn_attrs true")
            attr_t = gather_attr_targets(matched_idxs, labels, targets)
        if self.attr_loss == "ce":
            n_total = labels_cat.numel() if self.norm == "region" else None
            loss, log = L.attribute_ce_loss(attr_logits, attr_t, A.ACTIVATION, n_total=n_total)
            if loss is not None:
                out["loss_attr_ce"] = self.weights["attr"] * loss
            self.last_log.update(log)
        # every knowledge-base loss (each 1 - Sat): see ltn_logic.kb_losses
        kb, log = L.kb_losses(class_logits, labels_cat, attr_logits, attr_t or {}, A.VALUE_INDEX, p=self.ltn_p, activation=A.ACTIVATION,
                              class_loss=self.class_loss, class_agg=self.class_agg,
                              class_bg_weight=self.class_bg_weight,
                              use_attr_facts=self.attr_loss == "facts", axioms=self.axioms, ax_guard=self.ax_guard,
                              ax_family=self.ax_family, norm=self.norm)
        # lambda weights (all 1.0 in the v5 runs)
        for name, w in (("loss_class_sat", "class"), ("loss_find", "find"), ("loss_attr", "attr"),
                        ("loss_clin", "clin")):
            if name in kb:
                out[name] = self.weights[w] * kb[name]
        self.last_log.update(log)
        if self.aux:
            probs = L.head_probs(attr_logits, A.ACTIVATION)
            ax, log = L.aux_losses(probs, z.float(), getattr(self, "decoders", None), labels_cat, A.ACTIVATION,
                                   A.FAMILY, self.aux, ent_target=self.ent_target, norm=self.norm)
            for name, v in ax.items():
                out[name] = self.weights[name[len("loss_"):]] * v
            self.last_log.update(log)
        self.last_log["ltn_p"] = float(self.ltn_p)
        return out

    # ---------------------------------------------------------------------------------------------- inference
    def _postprocess(self, class_logits, box_regression, attr_logits, proposals, image_shapes):
        """torchvision's postprocess_detections, line by line, plus the index of the region behind every detection,
        so that the concept probabilities of that region can be returned with it."""
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [b.shape[0] for b in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        attr_probs = L.head_probs({h: v.float() for h, v in attr_logits.items()}, A.ACTIVATION)

        out = []
        start = 0
        for boxes, scores, image_shape, n in zip(pred_boxes.split(boxes_per_image, 0),
                                                 pred_scores.split(boxes_per_image, 0), image_shapes, boxes_per_image):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device).view(1, -1).expand_as(scores)
            # index of the region behind every (box, class) candidate, to return its concept probabilities after NMS
            region = torch.arange(start, start + n, device=device).view(-1, 1).expand_as(scores)
            boxes, scores, labels, region = boxes[:, 1:], scores[:, 1:], labels[:, 1:], region[:, 1:]
            boxes, scores = boxes.reshape(-1, 4), scores.reshape(-1)
            labels, region = labels.reshape(-1), region.reshape(-1)
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, region = boxes[inds], scores[inds], labels[inds], region[inds]
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, region = boxes[keep], scores[keep], labels[keep], region[keep]
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[: self.detections_per_img]
            det = {"boxes": boxes[keep], "labels": labels[keep], "scores": scores[keep]}
            for h, p in attr_probs.items():
                det[f"attr_{h}"] = p[region[keep]]
            out.append(det)
            start += n
        return out


def install(model, **kw):
    """Replace model.roi_heads with LTNRoIHeads, keeping the same modules (and their weights).
    Keywords as LTNRoIHeads: class_loss, class_agg, class_bg_weight, attr_loss, axioms, ax_guard, ax_family, aux,
    ent_target, weights, norm."""
    model.roi_heads = LTNRoIHeads(model.roi_heads, **kw)
    return model
