"""
BI-RADS facts of each lesion, for the LTN (thesis Chapter 6, "Facts"). Version 5 (call of 6 Nov 2026).

Version 5: the facts are written ONCE to a facts file (build_facts.py -> facts_v5.csv, one row per lesion of the train /
validation split, one column per fact phi1-phi21 of SymMedNet, appendix B.9.1), and train_frcnn.py reads the facts only
from that file (load_facts_file). The file is what the model sees, and can be checked by hand.

Changes of version 4 from version 3
  * no category head and no category facts: the knowledge base concludes benign / malignant directly (SymMedNet)
  * mass margin: sigmoid (multi-label) instead of softmax; a compound CBIS margin makes every listed value true
  * mass density: softmax with 2 outputs (high, low) instead of one sigmoid

Sources
  * Official CBIS-DDSM case-description CSVs (mass_case_description_*.csv, calc_case_description_*.csv), one row per
    lesion: shape, margin, morphology, distribution. A lesion of the manifest (image_id, lesion_id) is the CSV row
    (image folder, abnormality id). Only rows of images in our train / validation splits are kept (load_cbis_csvs):
    test rows are dropped on reading and never used.
  * The group's image-level file (cbis_ddsm_image_level_combined.xlsx) only for mass density, which the CSVs do not
    have. It is given per IMAGE, so it is attached only when the image has exactly one lesion.

Heads (sheet "Predicate heads" of LTN_axioms_v4.xlsx)
    head          family  activation  values
    mass_shape    MASS    softmax     oval, round, irregular
    mass_margin   MASS    sigmoid     circumscribed, obscured, microlobulated, indistinct, spiculated
    mass_density  MASS    softmax     high, low
    calc_morph    CALC    sigmoid     typically_benign, coarse_heterogeneous, pleomorphic, amorphous,
                                      fine_linear_branching
    calc_distr    CALC    softmax     clustered, segmental, linear, regional, diffuse

Encoding: per head a vector over its values, or None (no fact for that head).
  softmax head: one-hot; a compound CBIS value is first resolved to its MOST SUSPICIOUS value; values of equal
                suspicion stay multi-hot = disjunction.
  sigmoid head: multi-hot of every value present; the values NOT listed are false (CBIS lists every value), so the
                fact is, value by value, has(x, v) or not has(x, v).

Nothing here touches the model; it only prepares labels.
"""
from typing import Dict, Iterable, List, Optional, Tuple
import os

import numpy as np
import pandas as pd

VERSION = 5

# head -> (family, activation, values)
HEADS: Dict[str, tuple] = {
    "mass_shape":   ("MASS", "softmax", ["oval", "round", "irregular"]),
    "mass_margin":  ("MASS", "sigmoid", ["circumscribed", "obscured", "microlobulated", "indistinct", "spiculated"]),
    "mass_density": ("MASS", "softmax", ["high", "low"]),
    "calc_morph":   ("CALC", "sigmoid", ["typically_benign", "coarse_heterogeneous", "pleomorphic", "amorphous",
                                         "fine_linear_branching"]),
    "calc_distr":   ("CALC", "softmax", ["clustered", "segmental", "linear", "regional", "diffuse"]),
}
HEAD_NAMES: List[str] = list(HEADS)
HEAD_SIZES: Dict[str, int] = {h: len(v[2]) for h, v in HEADS.items()}
ACTIVATION: Dict[str, str] = {h: v[1] for h, v in HEADS.items()}
FAMILY: Dict[str, str] = {h: v[0] for h, v in HEADS.items()}
VALUE_INDEX: Dict[str, Dict[str, int]] = {h: {name: i for i, name in enumerate(v[2])} for h, v in HEADS.items()}

# ---------------------------------------------------------------------------------------- CBIS vocabulary
# CBIS token -> head value. Tokens in *_IGNORE are CBIS descriptors that are not values of the head (they leave the
# head without a fact if nothing else is listed). Any other token raises: no silent default.
SHAPE_MAP = {"OVAL": "oval", "LOBULATED": "oval",          # lobulated was merged into oval in BI-RADS 5th ed.
             "ROUND": "round", "IRREGULAR": "irregular"}
SHAPE_IGNORE = {"ARCHITECTURAL_DISTORTION", "LYMPH_NODE", "FOCAL_ASYMMETRIC_DENSITY", "ASYMMETRIC_BREAST_TISSUE"}
# order of suspicion used by most_suspicious (higher = more suspicious)
SHAPE_RANK = {"oval": 1, "round": 1, "irregular": 2}

# ILL_DEFINED is BI-RADS 'indistinct'
MARGIN_MAP = {"CIRCUMSCRIBED": "circumscribed", "OBSCURED": "obscured", "MICROLOBULATED": "microlobulated",
              "ILL_DEFINED": "indistinct", "SPICULATED": "spiculated"}

DISTR_MAP = {"CLUSTERED": "clustered", "SEGMENTAL": "segmental", "LINEAR": "linear", "REGIONAL": "regional",
             "DIFFUSELY_SCATTERED": "diffuse"}
DISTR_RANK = {"diffuse": 1, "regional": 2, "clustered": 3, "linear": 4, "segmental": 5}   # BI-RADS order of suspicion

# all BI-RADS 'typically benign' types are one value of the head (as SymMedNet's c_morph_t_ben)
MORPH_MAP = {"PLEOMORPHIC": "pleomorphic", "AMORPHOUS": "amorphous", "FINE_LINEAR_BRANCHING": "fine_linear_branching",
             **{t: "typically_benign" for t in ("PUNCTATE", "LUCENT_CENTER", "LUCENT_CENTERED", "VASCULAR", "COARSE",
                                                "ROUND_AND_REGULAR", "EGGSHELL", "DYSTROPHIC", "LARGE_RODLIKE",
                                                "SKIN", "MILK_OF_CALCIUM")}}

DENSITY_COLUMNS = ("mass_density_HIGH", "mass_density_LOW")          # group's file
CSV_COLUMNS = {"MASS": ("mass shape", "mass margins"), "CALC": ("calc type", "calc distribution")}
CSV_COMMON = ("image file path", "abnormality id", "abnormality type")


def _tokens(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    return [t for t in s.split("-") if t] if s and s.upper() != "N/A" else []


def _map_tokens(cell, mapping: dict, ignore: Iterable[str] = (), what: str = "") -> List[str]:
    out = []
    for t in _tokens(cell):
        if t in mapping:
            out.append(mapping[t])
        elif t not in ignore:
            raise KeyError(f"unknown CBIS {what} token {t!r} in {cell!r}")
    return out


def most_suspicious(values: List[str], head: str, rank: dict) -> Optional[List[float]]:
    """Single-valued head (softmax): keep the most suspicious value; ties stay multi-hot (disjunction)."""
    if not values:
        return None
    top = max(rank[v] for v in values)
    keep = {v for v in values if rank[v] == top}
    return [1.0 if v in keep else 0.0 for v in HEADS[head][2]]


def all_present(values: List[str], head: str) -> Optional[List[float]]:
    """Multi-label head (sigmoid): every value present is 1, every other value 0 (a negative fact)."""
    if not values:
        return None
    return [1.0 if v in set(values) else 0.0 for v in HEADS[head][2]]


# ---------------------------------------------------------------------------------------- CBIS CSVs
def family_of_csv_row(row) -> str:
    t = str(row["abnormality type"]).strip().lower()
    if t == "mass":
        return "MASS"
    if t == "calcification":
        return "CALC"
    raise KeyError(f"unexpected abnormality type {row['abnormality type']!r}")


def load_cbis_csvs(paths: Iterable[str], keep_image_ids: set) -> Tuple[Dict[Tuple[str, int], dict], Dict[str, int]]:
    """Read the CBIS case-description CSVs and index them by (image_id, abnormality id).
    Rows whose image is not in `keep_image_ids` (our train + validation images) are dropped at once and never
    looked at: this is how the locked test split stays untouched even if a test CSV is passed.
    Returns the index and, per file, how many rows were kept."""
    index, kept = {}, {}
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        fam_cols = [c for cols in CSV_COLUMNS.values() for c in cols if c in df.columns]
        missing = [c for c in CSV_COMMON if c not in df.columns]
        if missing or len(fam_cols) != 2:
            raise KeyError(f"{p}: not a CBIS case-description CSV (missing {missing or 'attribute columns'})")
        # 'image file path' starts with the image folder = our image_id
        df["image_id"] = df["image file path"].astype(str).str.split("/").str[0].str.strip()
        df = df[df["image_id"].isin(keep_image_ids)]                      # test rows dropped here
        kept[os.path.basename(p)] = int(len(df))
        for _, r in df.iterrows():
            key = (r["image_id"], int(r["abnormality id"]))
            if key in index:
                raise KeyError(f"lesion {key} appears twice in the CBIS CSVs")
            rec = {c: r[c] for c in CSV_COMMON + tuple(fam_cols)}
            rec["family"] = family_of_csv_row(r)
            index[key] = rec
    return index, kept


def csv_attributes(rec: dict) -> Dict[str, Optional[List[float]]]:
    """Facts from one CBIS CSV row: the heads of its family."""
    if rec["family"] == "MASS":
        return {"mass_shape": most_suspicious(_map_tokens(rec["mass shape"], SHAPE_MAP, SHAPE_IGNORE, "shape"),
                                              "mass_shape", SHAPE_RANK),
                "mass_margin": all_present(_map_tokens(rec["mass margins"], MARGIN_MAP, (), "margin"), "mass_margin")}
    return {"calc_morph": all_present(_map_tokens(rec["calc type"], MORPH_MAP, (), "calc type"), "calc_morph"),
            "calc_distr": most_suspicious(_map_tokens(rec["calc distribution"], DISTR_MAP, (), "distribution"),
                                          "calc_distr", DISTR_RANK)}


# ---------------------------------------------------------------------------------------- group's file (density)
def check_columns(columns) -> None:
    """Raise if the group's file lacks the density columns (never a silent default)."""
    missing = [c for c in DENSITY_COLUMNS if c not in set(columns)]
    if missing:
        raise KeyError(f"density columns missing from the file: {missing}")


def _flag(v) -> int:
    """One-hot cell -> 0/1. NaN counts as 0; anything else must be 0 or 1."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    iv = int(v)
    if iv not in (0, 1) or float(v) != iv:
        raise ValueError(f"one-hot cell with value {v!r}")
    return iv


def density_vector(xlsx_row) -> Optional[List[float]]:
    """[high, low] one-hot (softmax head with 2 outputs); None when neither or both are set."""
    high, low = _flag(xlsx_row["mass_density_HIGH"]), _flag(xlsx_row["mass_density_LOW"])
    if high == low:
        return None
    return [float(high), float(low)]


# ---------------------------------------------------------------------------------------- one lesion
def lesion_attributes(csv_rec: Optional[dict], xlsx_row, family: str, n_lesions_in_row: int
                      ) -> Optional[Dict[str, Optional[list]]]:
    """Facts of one lesion. csv_rec = its CBIS CSV row (None if not found: no CSV facts); density from the group's
    file only when the image has one lesion. Returns None when nothing at all is known.
    Plain lists, so the records stay JSON-serialisable."""
    if family not in ("MASS", "CALC"):
        raise KeyError(f"unexpected family {family!r}")
    if n_lesions_in_row < 1:
        raise ValueError("a lesion was asked for a row without lesions")
    out: Dict[str, Optional[list]] = {}
    if csv_rec is not None:
        if csv_rec["family"] != family:
            raise ValueError(f"family {family} of the image, {csv_rec['family']} in the CBIS CSV")
        out.update(csv_attributes(csv_rec))
    if family == "MASS":
        # density is given per image: attached only when the image has one lesion (else unknown)
        out["mass_density"] = density_vector(xlsx_row) if n_lesions_in_row == 1 else None
    if all(v is None for v in out.values()):
        return None
    return out


def target_tensors(lesion_attrs: List[Optional[dict]]):
    """For the lesions of one image, in the order of their boxes, build per head:
         attr_<head>       float32 [n, K]  vector of the fact (zeros where unknown)
         attr_<head>_valid bool    [n]     True where the head has a fact for that lesion
    Lesions of the other family and heads without a fact are 'not valid'.
    Returns numpy arrays (the Dataset converts them to tensors)."""
    n = len(lesion_attrs)
    out = {}
    for head, k in HEAD_SIZES.items():
        y = np.zeros((n, k), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        for i, la in enumerate(lesion_attrs):
            if la is None:
                continue
            v = la.get(head)
            if v is not None:
                if len(v) != k:
                    raise ValueError(f"head {head}: fact of length {len(v)}, expected {k} (records of an older "
                                     f"version? rebuild them)")
                y[i] = np.asarray(v, dtype=np.float32)
                valid[i] = True
        out[f"attr_{head}"] = y
        out[f"attr_{head}_valid"] = valid
    return out


def describe(lesion_attrs: List[Optional[dict]]) -> Dict[str, Dict[str, int]]:
    """Counts for a printout: per head, lesions with a fact; for softmax heads how many are a disjunction (ties),
    for sigmoid heads how many have more than one value."""
    rep = {}
    for head in HEAD_NAMES:
        known = [la[head] for la in lesion_attrs if la is not None and la.get(head) is not None]
        rep[head] = dict(known=len(known), multi=int(sum(np.asarray(v).sum() > 1 for v in known)))
    return rep


# ---------------------------------------------------------------------------------------- facts file (version 5)
# One column per fact of SymMedNet (appendix B.9.1), in the paper's order. Cell value: 1 (fact true), 0 (false: for a
# sigmoid head the value is not listed, for a softmax head another value is), empty = no fact (unknown).
# phi1 / phi2 (has_mass / has_calc) are the family of the lesion; in the detector they are computed from the class head.
FINDING_COLUMNS = {"phi01_has_mass": "MASS", "phi02_has_calc": "CALC"}
FACT_COLUMNS = [   # (column, head, value)
    ("phi03_density_high", "mass_density", "high"),
    ("phi03_density_low", "mass_density", "low"),      # second output of the 2-way density head (not in the paper)
    ("phi04_shape_oval", "mass_shape", "oval"),
    ("phi05_shape_round", "mass_shape", "round"),
    ("phi06_shape_irregular", "mass_shape", "irregular"),
    ("phi07_margin_circumscribed", "mass_margin", "circumscribed"),
    ("phi08_margin_obscured", "mass_margin", "obscured"),
    ("phi09_margin_microlobulated", "mass_margin", "microlobulated"),
    ("phi10_margin_indistinct", "mass_margin", "indistinct"),
    ("phi11_margin_spiculated", "mass_margin", "spiculated"),
    ("phi12_morph_typically_benign", "calc_morph", "typically_benign"),
    ("phi13_morph_coarse_heterogeneous", "calc_morph", "coarse_heterogeneous"),
    ("phi14_morph_pleomorphic", "calc_morph", "pleomorphic"),
    ("phi15_morph_amorphous", "calc_morph", "amorphous"),
    ("phi16_morph_fine_linear_branching", "calc_morph", "fine_linear_branching"),
    ("phi17_distr_clustered", "calc_distr", "clustered"),
    ("phi18_distr_segmental", "calc_distr", "segmental"),
    ("phi19_distr_linear", "calc_distr", "linear"),
    ("phi20_distr_regional", "calc_distr", "regional"),
    ("phi21_distr_diffuse", "calc_distr", "diffuse"),
]
ID_COLUMNS = ["image_id", "lesion_id", "split", "family", "pathology", "n_lesions_image", "in_cbis_csv",
              "cbis_shape", "cbis_margin", "cbis_calc_type", "cbis_distribution", "density_source"]
SPLITS_ALLOWED = ("train", "val")


def facts_row(image_id: str, lesion_id: int, split: str, family: str, pathology: str, n_lesions: int,
              csv_rec: Optional[dict], xlsx_row) -> dict:
    """One row of the facts file: the identifiers, the raw CBIS strings, and one column per fact."""
    if split not in SPLITS_ALLOWED:
        raise ValueError(f"{image_id}: split {split!r} - the facts file holds train / val lesions only")
    attrs = lesion_attributes(csv_rec, xlsx_row, family, n_lesions) or {}
    row = dict(image_id=image_id, lesion_id=int(lesion_id), split=split, family=family, pathology=pathology,
               n_lesions_image=int(n_lesions), in_cbis_csv=csv_rec is not None,
               cbis_shape=(csv_rec or {}).get("mass shape", ""), cbis_margin=(csv_rec or {}).get("mass margins", ""),
               cbis_calc_type=(csv_rec or {}).get("calc type", ""),
               cbis_distribution=(csv_rec or {}).get("calc distribution", ""),
               density_source="group file (single-lesion image)" if attrs.get("mass_density") is not None else "")
    for col, fam in FINDING_COLUMNS.items():
        row[col] = int(family == fam)
    # 1 / 0 = fact true / false, empty cell = no fact for that head
    for col, head, value in FACT_COLUMNS:
        v = attrs.get(head)
        row[col] = "" if v is None else int(v[VALUE_INDEX[head][value]])
    for k in ("cbis_shape", "cbis_margin", "cbis_calc_type", "cbis_distribution"):
        if isinstance(row[k], float) and np.isnan(row[k]):
            row[k] = ""
    return row


def load_facts_file(path: str) -> Dict[Tuple[str, int], dict]:
    """Read facts_v5.csv. Returns {(image_id, lesion_id): {"family": ..., "split": ..., "attrs": {head: vector|None}}}.
    Raises if the file has a test row, a missing column, a lesion twice, or a fact that is not 0 / 1 / empty, or if a
    softmax head is known but has no true value."""
    df = pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False)
    need = ID_COLUMNS + list(FINDING_COLUMNS) + [c for c, _, _ in FACT_COLUMNS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: columns missing {missing} (rebuild it with build_facts.py version 5)")
    bad_split = sorted(set(df["split"]) - set(SPLITS_ALLOWED))
    if bad_split:
        raise ValueError(f"{path}: rows with split {bad_split}: the test split must never be in the facts file")
    out = {}
    for _, r in df.iterrows():
        key = (r["image_id"], int(r["lesion_id"]))
        if key in out:
            raise KeyError(f"{path}: lesion {key} twice")
        fam = r["family"]
        if fam not in ("MASS", "CALC"):
            raise ValueError(f"{path}: lesion {key} with family {fam!r}")
        attrs: Dict[str, Optional[list]] = {}
        for head in HEAD_NAMES:
            cols = [(c, v) for c, h, v in FACT_COLUMNS if h == head]
            cells = [str(r[c]).strip() for c, _ in cols]
            if all(x == "" for x in cells):
                attrs[head] = None
                continue
            if any(x not in ("0", "1") for x in cells):
                raise ValueError(f"{path}: lesion {key}, head {head}: values {cells} (expected 0, 1 or all empty)")
            # rebuild the head's vector from its columns, in the order of HEADS
            vec = [0.0] * HEAD_SIZES[head]
            for (c, v), x in zip(cols, cells):
                vec[VALUE_INDEX[head][v]] = float(x)
            if ACTIVATION[head] == "softmax" and sum(vec) == 0:
                raise ValueError(f"{path}: lesion {key}, softmax head {head} known but no value is 1")
            if FAMILY[head] != fam:
                raise ValueError(f"{path}: lesion {key} ({fam}) has facts for the {FAMILY[head]} head {head}")
            attrs[head] = vec
        out[key] = dict(family=fam, split=r["split"], attrs=attrs if any(v is not None for v in attrs.values()) else None)
    return out
