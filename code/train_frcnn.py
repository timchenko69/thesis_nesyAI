r"""Training of Faster R-CNN ResNet-50 FPN with 5 classes on CBIS-DDSM (same code as notebooks 08 and 11).

Generated from the cells of notebook 11. Compared with the notebook:
* parameters from the command line (--name value) or from a JSON file (--config file.json);
* option `normalization`: "imagenet" (PNG / 65535, then ImageNet mean and std) or "zscore";
* option `lr_schedule`: "step" (default: lr x lr_gamma every lr_step_size epochs, as the baseline of notebook 02) or
  "constant" (as all runs up to 10 October);
* last.pt also contains the optimiser, the GradScaler and the state of the random generators; --resume true resumes
  from there;
* check of the free GPU memory before starting;
* with W&B on, the script itself is uploaded to the run (Code tab as a code artifact, Files tab) and its SHA-256 goes
  into the summary (script_sha256);
* run_manifest.json in the run folder: configuration, command, environment, data, actual model, optimiser (type and
  hyper-parameters; the full state is in last.pt), results and files with SHA-256; updated at every epoch.
  --config <run_manifest.json> relaunches the same configuration;
* always runs with the mammoltn conda environment (F:\conda_envs\mammoltn): if started with another Python it
  relaunches itself with that one; alternatively use train_frcnn.bat, which calls it directly;
* no plots (sections 8a/8b of the notebook are skipped);
* cache_images (default false; --cache_images true to enable): training and validation images are read and resized
  once at the start and kept in memory (about 2 GB at 512 px); flip and normalisation still happen at every batch, so
  the data seen by the network are identical. Much faster: reading the PNGs is the bottleneck (with batch 32 about
  540 s of training per epoch without cache against 55 s with the cache);
* scaling + rotation augmentation (section 7b): --affine_p, --rotate_deg, --scale_range; off by default (affine_p 0).

Default values = SGD_step15_full (run
frcnn_r50fpn_5way_512_anc16-256_rpnfg05_sgd5e-3_step15_wd5e-4_bs4_tbl5_e25_imagenet_script, W&B gdn4iavp, 10 October
2026): SGD lr 0.005, momentum 0.9, weight decay 5e-4, lr x0.1 every 15 epochs, batch 4, 25 epochs, whole network
trainable, normalisation /65535 + ImageNet, anchors 16-256, RPN thresholds 0.5/0.3, 512 px, seed 1337. Only
difference: cache_images false (the data seen by the network do not change). Change only run_name (new folder).
SGD_step10_full (W&B q67mn9ir: step at epoch 10, 30 epochs):
    ... train_frcnn.py --lr_step_size 10 --epochs 30 --run_name <new name>
Experiment C (constant SGD 0.02, batch 32, z-score):
    ... train_frcnn.py --lr 0.02 --lr_schedule constant --batch_size 32 --normalization zscore --min_free_gpu_gb 20
        --run_name <new name>
Experiment D (Adam 1e-4, batch 16, anchors 32-512, threshold 0.7, ImageNet normalisation):
    ... train_frcnn.py --optimizer adam --lr 1e-4 --weight_decay 0.0 --batch_size 16 --normalization imagenet
        --anchor_sizes "((32,), (64,), (128,), (256,), (512,))" --rpn_fg_iou 0.7 --min_free_gpu_gb 12
        --run_name frcnn_r50fpn_5way_512_adam1e-4_bs16_imagenet
Augmentation (same configuration as gdn4iavp, plus scaling + rotation):
    ... train_frcnn.py --config F:\runs\<gdn4iavp folder>\run_manifest.json --resume false --affine_p 0.5
        --run_name <new name>

Examples (from C:\Users\grains7lab2\Desktop\Tesi_Karimov\new_code):
    F:\conda_envs\mammoltn\python.exe train_frcnn.py
    F:\conda_envs\mammoltn\python.exe train_frcnn.py --normalization zscore --run_name frcnn_..._zscore
    F:\conda_envs\mammoltn\python.exe train_frcnn.py --resume true            (resumes the same run_name)
    F:\conda_envs\mammoltn\python.exe train_frcnn.py --config experiment.json
    F:\conda_envs\mammoltn\python.exe train_frcnn.py --config F:\runs\<run>\run_manifest.json --run_name <new name>
To save the output: append  > log.txt 2>&1
"""
import os as _os0, subprocess as _sp0, sys as _sys0

# ---- environment: the script always runs in the mammoltn conda environment -----------------------------------------
# If it is started with another Python (terminal, VS Code, ...), it relaunches itself with mammoltn's Python, passing
# the same options. The path can be changed with the FRCNN_PYTHON environment variable.
MAMMOLTN_PYTHON = _os0.environ.get("FRCNN_PYTHON", r"F:\conda_envs\mammoltn\python.exe")
_same_env = (_os0.path.normcase(_os0.path.abspath(_sys0.executable))
             == _os0.path.normcase(_os0.path.abspath(MAMMOLTN_PYTHON)))
if not _same_env:
    if _os0.environ.get("FRCNN_RELAUNCHED"):
        raise RuntimeError(f"rilancio non riuscito: in esecuzione con {_sys0.executable}, atteso {MAMMOLTN_PYTHON}")
    if not _os0.path.exists(MAMMOLTN_PYTHON):
        raise FileNotFoundError(f"Python dell'ambiente mammoltn non trovato: {MAMMOLTN_PYTHON} "
                                f"(impostare FRCNN_PYTHON con il percorso corretto)")
    print(f"[ambiente] avviato con {_sys0.executable}: rilancio con {MAMMOLTN_PYTHON}", flush=True)
    _env = dict(_os0.environ, FRCNN_RELAUNCHED="1")
    _sys0.exit(_sp0.call([MAMMOLTN_PYTHON, _os0.path.abspath(__file__)] + _sys0.argv[1:], env=_env))
print(f"[ambiente] {_sys0.executable}", flush=True)

import matplotlib
matplotlib.use("Agg")      # no window: the script does not show plots
import argparse, ast, dataclasses, json as _json, os as _os, subprocess, sys as _sys


def parse_overrides(cfg_cls):
    """--field value for every Config field, or --config file.json. Values are Python literals
    (numbers, True/False, tuples, None); anything else stays a string."""
    p = argparse.ArgumentParser(description="Faster R-CNN 5 classi su CBIS-DDSM")
    p.add_argument("--config", help="file JSON con i valori da cambiare")
    fields = {f.name: f for f in dataclasses.fields(cfg_cls)}
    for name in fields:
        p.add_argument(f"--{name}")
    a = p.parse_args()
    out = {}
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            loaded = _json.load(f)
        if isinstance(loaded, dict) and "schema" in loaded and "config" in loaded:   # run_manifest.json of a run
            loaded = {k: v for k, v in loaded["config"].items() if k != "resume"}
        out.update(loaded)
    for name, f in fields.items():
        v = getattr(a, name)
        if v is None:
            continue
        if f.type in (bool, "bool"):
            low = v.strip().lower()
            if low not in ("true", "false", "1", "0"):
                raise ValueError(f"--{name}: usa true o false")
            out[name] = low in ("true", "1")
            continue
        try:
            out[name] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[name] = v
    unknown = set(out) - set(fields)
    if unknown:
        raise ValueError(f"campi sconosciuti: {sorted(unknown)}")
    for k in ("anchor_sizes", "anchor_ratios", "froc_fp_points"):
        if k in out and isinstance(out[k], list):
            out[k] = tuple(tuple(x) if isinstance(x, list) else x for x in out[k])
    return out


def gpu_free_gb(index=0):
    """Free memory according to nvidia-smi (on Windows torch.cuda.mem_get_info does not see other processes)."""
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
                               "-i", str(index)], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        used, total = (float(x) for x in line.split(","))
        return (total - used) / 1024, total / 1024
    except Exception:
        import torch
        free, total = torch.cuda.mem_get_info(index)
        return free / 2 ** 30, total / 2 ** 30


# ====================================================================================================
# 1. Import, seed, device
# ====================================================================================================
import os, sys, re, json, math, time, random, hashlib, datetime
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.rpn import RPNHead


print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__)
print("torchvision", torchvision.__version__)
DEVICE = torch.device(os.environ.get("FRCNN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
print("device     ", DEVICE)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ====================================================================================================
# 2. Configuration — every choice is here
# ====================================================================================================
@dataclass
class Config:
    # --- original data (the same as 01_DAT) ---
    data_root: str = r"F:\MammoLTN-full-images"
    png_dir_name: str = "CBIS-DDSM-PNG16-CROPPED-MASKS"
    splits_dir_name: str = "CBIS_08_11_2024"
    combined_xlsx_name: str = "cbis_ddsm_image_level_combined.xlsx"

    # --- geometry ---
    image_size: int = 512                 # long side after resize
    normalization: str = "imagenet"       # "imagenet": PNG / maximum value (65535) -> [0, 1], then ImageNet mean and
                                          # std inside the model (as notebooks 02 and 05-07);
                                          # "zscore": per-image z-score (experiments A, B, C, B_script)

    # --- model ---
    pretrained: bool = True               # COCO weights, then a 5-class head
    trainable_backbone_layers: int = 5    # the whole network (user's choice, as in Famouri)

    # --- RPN: anchors and thresholds (notebook 09) ---
    # one tuple of sizes for each of the 5 FPN levels (P2..P6); torchvision default:
    # ((32,), (64,), (128,), (256,), (512,)). With several sizes per level, e.g.
    # ((16, 20.2, 25.4), (32, 40.3, 50.8), ...), the RPN head is adapted by build_model().
    anchor_sizes: tuple = ((16,), (32,), (64,), (128,), (256,))
    anchor_ratios: tuple = (0.5, 1.0, 2.0)   # height/width, the same on every level
    rpn_fg_iou: float = 0.5               # anchor is positive if IoU >= (torchvision default 0.7)
    rpn_bg_iou: float = 0.3               # anchor is negative if IoU <  (torchvision default 0.3)

    # --- optimisation (default: SGD_step15_full = recipe of baseline 02, whole network, step at 15) ---
    optimizer: str = "sgd"                # "sgd" or "adam" (experiments A, B, B_script, B_imagenet)
    lr: float = 5e-3                      # initial lr; C: 0.02 constant; Adam runs: 1e-5
    momentum: float = 0.9                 # SGD only
    weight_decay: float = 5e-4            # Adam runs: 0
    lr_schedule: str = "step"             # "step": lr x lr_gamma every lr_step_size epochs; "constant": fixed lr
    lr_step_size: int = 15                # "step" only (SGD_step15_full: 15; baseline 02 and SGD_step10_full: 10)
    lr_gamma: float = 0.1                 # "step" only (baseline 02: 0.1)
    batch_size: int = 4                   # 455 steps per epoch; C and Adam runs: 32
    epochs: int = 25                      # SGD_step15_full: 25; SGD_step10_full: 30
    patience: int = 0                     # early stopping: stop when val mAP@50 has not improved for this many
                                          # epochs (best.pt is kept); 0 = off, train all `epochs`
    amp: bool = True
    grad_clip: Optional[float] = None

    # --- data ---
    hflip_p: float = 0.5                  # horizontal flip
    affine_p: float = 0.0                 # probability of scaling + rotation per image (0 = off, as in gdn4iavp)
    rotate_deg: float = 15.0              # maximum angle in degrees, [-rotate_deg, +rotate_deg] (Abdikenov 2025: ±15°)
    scale_range: tuple = (0.9, 1.1)       # zoom factor (Abdikenov 2025: ±10%); (1.0, 1.0) = no zoom
    clahe: bool = False                   # CLAHE on EVERY image, train and validation: preprocessing, not
                                          # augmentation (Abdikenov 2025: clip 2.0, tile 8x8); False = as gdn4iavp
    clahe_clip: float = 2.0               # clipLimit of cv2.createCLAHE
    clahe_grid: int = 8                   # tiles per side: tileGridSize = (clahe_grid, clahe_grid)
    photo_p: float = 0.0                  # probability of photometric jitter per training image (0 = off, as gdn4iavp)
    photo_contrast: tuple = (0.8, 1.2)    # contrast factor, uniform in this range (SymBreastNet §4.2)
    photo_gamma: tuple = (0.5, 0.75, 1.0) # gamma, one value drawn from this set (SymBreastNet §4.2 lists
                                          # {0, 0.5, 0.75, 1}: 0 is left out, x**0 = 1 would erase the image)
    # --- second round of augmentation (Oza et al. 2022, J. Imaging 8:141; Garcea et al. 2023). Every one is off
    #     by default, and with its probability at 0 the random stream is the same as before ---
    vflip_p: float = 0.0                  # vertical flip (Oza 2022, Table 2: horizontal + vertical flips with
                                          # Faster R-CNN on CBIS-DDSM; Ribli 2018)
    noise_p: float = 0.0                  # probability of Gaussian noise on the breast pixels
    noise_std: float = 0.01               # maximum std on the [0, 1] scale; each image draws sigma ~ U(0, noise_std)
    erase_p: float = 0.0                  # probability of random erasing (Cutout), only away from the lesion boxes
    erase_area: tuple = (0.01, 0.05)      # area of the erased rectangle, as a fraction of the image area
    elastic_p: float = 0.0                # probability of elastic deformation (Cao 2019 in Oza 2022; Garcea 2023)
    elastic_grid: int = 4                 # control points per side of the coarse displacement grid
    elastic_px: float = 8.0               # maximum displacement of a control point, in pixels (long side 512)

    # --- LTN (thesis Ch. 6), version 5 (call of 6 Nov 2026). Off by default: with False nothing below changes ---
    ltn_attrs: bool = False               # attach the BI-RADS facts phi3-phi21 to each lesion, read ONLY from the
                                          # facts file below, as attr_<head> / attr_<head>_valid in the targets
    ltn_facts_file: str = ""              # facts_v5.csv written by build_facts.py (one row per train/val lesion)
    ltn_csv_dir: str = ""                 # used by build_facts.py only (kept so that old configs still load)
    ltn_csv_files: tuple = ("mass_case_description_train_set.csv", "calc_case_description_train_set.csv")
    ltn: bool = False                     # replace the RoI heads with LTNRoIHeads (ltn_heads.py); alone = the baseline
    ltn_class_loss: str = "ce"            # phi_class: "ce" (torchvision's cross-entropy, the baseline) | "sat" (1 - Sat)
    ltn_class_agg: str = "geo"            # aggregator of phi_class with "sat": "geo" (geometric mean = product,
                                          # 1 - Sat = 1 - exp(-CE)) | "balanced" (ApME, background ltn_class_bg_w)
                                          # | "all" (ApME, one clause) | "per_class" (ApME per class): see ltn_logic
    ltn_class_bg_w: float = 0.5           # weight of the background clause with ltn_class_agg "balanced"
    ltn_attr_loss: str = "none"           # concept heads: "none" | "facts" (phi1-phi21 as 1 - Sat: the "C" runs)
                                          # | "ce" (BC = R50+C, cross-entropy, no logic)
    ltn_axioms: tuple = ()                # axioms psi of ltn_logic.AXIOMS, or a preset of ltn_logic.AXIOM_SETS:
                                          # "kb16" = psi3-psi14, psi16 (all of SymMedNet without the Normal ones)
    ltn_ax_guard: str = "class"           # forall Diag(x, l_k) over the regions of class k (SymMedNet B.9) | "fg" | "all"
    ltn_ax_family: str = "pred"           # "pred": has_mass(x, c_mass) predicted, as in the paper | "label": the
                                          # family fixed by the region's label (see ltn_logic.axiom_truths)
    ltn_aux: tuple = ()                   # SymMedNet "+Aux" regularisers: any of ("ent", "kl", "rec")
    ltn_ent_target: float = 0.7           # entropy target, fraction of log C (SymMedNet B.5)
    ltn_w_class: float = 1.0              # weights (lambda, SymMedNet: all 1) of loss_class_sat, loss_find,
    ltn_w_find: float = 1.0               # loss_attr (or loss_attr_ce), loss_clin, loss_ent, loss_kl, loss_rec
    ltn_w_attr: float = 1.0
    ltn_w_clin: float = 1.0
    ltn_w_ent: float = 1.0
    ltn_w_kl: float = 1.0
    ltn_w_rec: float = 1.0
    ltn_part: bool = False                # older versions only (part-whole M1, M2 removed): must stay False
    ltn_w_part: float = 1.0               # older versions only: must stay 1.0
    ltn_w_cat: float = 1.0                # v3 only (category head, removed): must stay 1.0
    ltn_ax_domain: str = "all"            # v3 only (replaced by ltn_ax_guard): must stay "all"
    ltn_norm: str = "mean"                # "mean" | "region": each concept-fact / axiom / aux term scaled by the
                                          # share of sampled regions in its domain, like torchvision's box loss
    ltn_p: float = 0.0                    # ApME exponent: 0 = schedule 2 -> 10 (SymMedNet); > 0 = fixed
    num_workers: int = 0                  # this script supports only 0 (see the check below)
    cache_images: bool = False            # True: images already resized, kept in memory (read once, in
                                          # parallel, about 2 GB); much faster, same data
    cache_threads: int = 8                # threads for the initial read
    log_every: int = 10                  # steps between two log lines; 0 = silent

    # --- evaluation ---
    eval_every: int = 1
    froc_iou: float = 0.50
    froc_fp_points: tuple = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    # --- examples and diagnostics ---
    n_examples: int = 8
    n_mask_checks: int = 6

    # --- output ---
    out_root: str = r"F:\runs"
    #run_name: str = "frcnn_r50fpn_5way_512_C1_adam1e5_bs1_zscore"   # NEW folder, never reused
    #run_name: str = "frcnn_r50fpn_5way_512_anc16-256_rpnfg05_sgd2e-2_m09_wd5e-4_bs32_zscore_script"   # experiment C
    #run_name: str = "frcnn_r50fpn_5way_512_anc16-256_rpnfg05_sgd5e-3_step10_wd5e-4_bs4_tbl5_imagenet_rep"   # SGD_step10_full
    run_name: str = "frcnn_r50fpn_5way_512_anc16-256_rpnfg05_sgd5e-3_step15_wd5e-4_bs4_tbl5_e25_imagenet_rep"   # NEW folder
    # (SGD_step15_full uses the same name with "_script" instead of "_rep": its folder must not be reused)

    seed: int = 1337

    # --- Weights & Biases (optional) ---
    use_wandb: bool = True
    wandb_project: str = "nesy-mammography"
    wandb_entity: Optional[str] = None

    # --- script ---
    resume: bool = False                  # True: resume from last.pt (model, optimiser, scaler, random generators)
    min_free_gpu_gb: float = 8.0          # stop if the GPU has less free memory (batch 4: a few GB; batch 32: 20)


cfg = Config(**parse_overrides(Config))
# LTN (thesis Ch. 6): imported only when switched on; with ltn / ltn_attrs false this script is the baseline
if cfg.ltn_attrs:
    import ltn_attributes as LTNA          # file next to this script
if cfg.ltn:
    import ltn_heads as LTNH               # files next to this script
    import ltn_logic as LTNL
if cfg.num_workers != 0:
    raise ValueError("num_workers > 0 non è supportato da questo script (su Windows i processi figli"
                     " rieseguirebbero tutto il file): usa 0")
if DEVICE.type == "cuda":
    _free, _total = gpu_free_gb()
    print(f"GPU libera {_free:.1f} GB su {_total:.1f} GB")
    if _free < cfg.min_free_gpu_gb:
        raise RuntimeError(f"GPU con solo {_free:.1f} GB liberi (richiesti {cfg.min_free_gpu_gb}): "
                           f"probabilmente un altro training è in corso. Riduci --min_free_gpu_gb solo se sei sicuro.")
for k, v in asdict(cfg).items():
    print(f"  {k:<28} {v}")

# ====================================================================================================
# 2. Configuration — every choice is here
# ====================================================================================================
set_all_seeds(cfg.seed)

PNG_ROOT = os.path.join(cfg.data_root, cfg.png_dir_name)
IMAGES_DIR = os.path.join(PNG_ROOT, "images")
LESION_MASK_DIR = os.path.join(PNG_ROOT, "lesion_masks")
MANIFEST_CSV = os.path.join(PNG_ROOT, "manifest.csv")
SPLITS_XLSX = os.path.join(cfg.data_root, cfg.splits_dir_name, cfg.combined_xlsx_name)

RUN_DIR = os.path.join(cfg.out_root, cfg.run_name)
if os.path.isdir(RUN_DIR) and any(f.endswith(".pt") for f in os.listdir(RUN_DIR)) and not cfg.resume:
    raise FileExistsError(f"{RUN_DIR} contiene già checkpoint. Ogni configurazione usa una "
                          f"cartella nuova: cambia cfg.run_name.")
os.makedirs(RUN_DIR, exist_ok=True)
HISTORY_JSON = os.path.join(RUN_DIR, "history.json")
BEST_PT = os.path.join(RUN_DIR, "best.pt")
LAST_PT = os.path.join(RUN_DIR, "last.pt")

if cfg.resume and not os.path.exists(os.path.join(RUN_DIR, "last.pt")):
    raise FileNotFoundError(f"--resume: {RUN_DIR} non contiene last.pt")
_cfg_name = "config.json" if not cfg.resume else f"config_resume_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
with open(os.path.join(RUN_DIR, _cfg_name), "w", encoding="utf-8") as f:
    json.dump(asdict(cfg), f, indent=2)

FACTS_FILES: List[str] = []
if cfg.ltn_attrs:
    if not cfg.ltn_facts_file:
        raise ValueError("--ltn_attrs true richiede --ltn_facts_file (facts_v5.csv di build_facts.py)")
    FACTS_FILES = [cfg.ltn_facts_file]

for p in (IMAGES_DIR, LESION_MASK_DIR, MANIFEST_CSV, SPLITS_XLSX, *FACTS_FILES):
    print(("ok      " if os.path.exists(p) else "MANCANTE"), p)
if FACTS_FILES and not os.path.exists(FACTS_FILES[0]):
    raise FileNotFoundError(f"file dei fatti mancante: {FACTS_FILES[0]} (crealo con build_facts.py)")
print("run dir:", RUN_DIR)

# ====================================================================================================
# 3. Classes, split and manifest
# ====================================================================================================
CLASS_NAMES = {0: "background",
               1: "mass_benign", 2: "mass_malignant",
               3: "calc_benign", 4: "calc_malignant"}
NUM_CLASSES = len(CLASS_NAMES)
FOREGROUND_IDS = [1, 2, 3, 4]
LABEL2FAMILY = {1: "MASS", 2: "MASS", 3: "CALC", 4: "CALC"}
LABEL2PATHOLOGY = {1: "BENIGN", 2: "MALIGNANT", 3: "BENIGN", 4: "MALIGNANT"}
MALIGNANT_IDS = {2, 4}
_FP2LABEL = {("MASS", "BENIGN"): 1, ("MASS", "MALIGNANT"): 2,
             ("CALC", "BENIGN"): 3, ("CALC", "MALIGNANT"): 4}


def lesion_label(family: str, pathology: str) -> int:
    k = (str(family).strip().upper(), str(pathology).strip().upper())
    if k not in _FP2LABEL:
        raise KeyError(f"nessuna classe per {k}; attese {sorted(_FP2LABEL)}")
    return _FP2LABEL[k]


def family_from_image_id(image_id: str) -> str:
    if image_id.startswith("Mass"):
        return "MASS"
    if image_id.startswith("Calc"):
        return "CALC"
    raise KeyError(f"image_id {image_id!r} non inizia con 'Mass' né con 'Calc'")


_BASE_RE = re.compile(r"^(Mass|Calc)-(Training|Test)_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)$")


def parse_image_id(image_id: str) -> dict:
    """Mass-Training_P_00041_LEFT_CC -> family, patient, side, view."""
    m = _BASE_RE.match(image_id)
    if not m:
        raise ValueError(f"image_id {image_id!r} non ha il formato atteso "
                         f"'<Mass|Calc>-<Training|Test>_P_xxxxx_<LEFT|RIGHT>_<CC|MLO>'")
    return dict(family=m.group(1).upper(), patient=m.group(3),
                side=m.group(4), view=m.group(5),
                base_key=f"{m.group(3)}_{m.group(4)}_{m.group(5)}")


def image_id_from_filename(fn: str) -> str:
    fn = str(fn)
    return fn[:-9] if fn.endswith("_crop.png") else os.path.splitext(fn)[0]


def image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}_crop.png")


def mask_path(image_id: str, lesion_id) -> str:
    return os.path.join(LESION_MASK_DIR, f"{image_id}_{int(lesion_id)}_mask.png")


def _is_mixed_pathology(source_patch_labels) -> bool:
    s = str(source_patch_labels)
    return ("BENIGN" in s) and ("MALIGNANT" in s)


def load_splits() -> Dict[str, pd.DataFrame]:
    df = pd.read_excel(SPLITS_XLSX)
    df.columns = [c.strip() for c in df.columns]
    df["image_id"] = df["image_filename"].map(image_id_from_filename)
    df["label"] = df["label_upd"].astype(str).str.strip().str.upper()
    mixed = df["source_patch_labels"].map(_is_mixed_pathology)
    print(f"escluse {int(mixed.sum())} immagini con patologia mista")
    df = df[~mixed].copy()
    split_map = {"Train": "train", "Valid": "val", "Test": "test"}
    unknown = sorted(set(df["split"].unique()) - set(split_map))
    if unknown:
        raise KeyError(f"valori di split inattesi: {unknown}")
    df["split_norm"] = df["split"].map(split_map)
    return {s: df[df["split_norm"] == s].reset_index(drop=True) for s in ("train", "val")}


splits = load_splits()
for s, d in splits.items():
    print(f"{s}: {len(d)} righe  {dict(Counter(family_from_image_id(i) for i in d['image_id']))}")
print("test: NON caricato (bloccato)")

# LTN: BI-RADS facts per lesion, key (image_id, lesion_id)
FACTS: Dict[Tuple[str, int], dict] = {}
if cfg.ltn_attrs:
    # facts per lesion, read only from the facts file (train / val rows only: load_facts_file refuses test rows)
    FACTS = LTNA.load_facts_file(cfg.ltn_facts_file)
    _split_of = {i: s for s, d in splits.items() for i in d["image_id"]}
    _wrong = [(k, v["split"], _split_of.get(k[0])) for k, v in FACTS.items()
              if k[0] in _split_of and v["split"] != _split_of[k[0]]]
    if _wrong:
        raise ValueError(f"facts file and split file disagree on {len(_wrong)} lesions, e.g. {_wrong[:3]}: rebuild it")
    print(f"file dei fatti: {len(FACTS)} lesioni ({sum(v['attrs'] is not None for v in FACTS.values())} con fatti) "
          f"<- {cfg.ltn_facts_file}")

manifest = pd.read_csv(MANIFEST_CSV)
manifest.columns = [c.strip() for c in manifest.columns]
manifest = manifest[manifest["status"].astype(str).str.lower() == "ok"]
manifest_by_img = {k: v for k, v in manifest.groupby("image_id")}
print("manifest (status=ok):", len(manifest), "righe lesione su", len(manifest_by_img), "immagini")

# ====================================================================================================
# 4. Records, with the mask ↔ image check
# ====================================================================================================
def png_hw(path: str) -> Tuple[int, int]:
    """(height, width) read from the header, without decoding the pixels."""
    with Image.open(path) as im:
        w, h = im.size
    return int(h), int(w)


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def build_records(split_df: pd.DataFrame, split: str,
                  mismatches: List[dict], qc: Dict[str, list]) -> List[dict]:
    records = []
    reasons = Counter()
    for _, row in split_df.iterrows():
        image_id = row["image_id"]
        info = parse_image_id(image_id)
        family = info["family"]
        pathology = str(row["label"]).strip().upper()
        if pathology not in ("BENIGN", "MALIGNANT"):
            raise KeyError(f"{image_id}: etichetta inattesa {pathology!r}")
        lab = lesion_label(family, pathology)

        ip = image_path(image_id)
        if not os.path.exists(ip):
            reasons["immagine mancante"] += 1
            continue
        mrows = manifest_by_img.get(image_id)
        if mrows is None:
            reasons["non nel manifest"] += 1
            continue
        img_hw = png_hw(ip)

        lesions = []
        for _, mrow in mrows.iterrows():
            lid = mrow.get("lesion_id")
            if pd.isna(lid):
                continue
            mp = mask_path(image_id, lid)
            if not os.path.exists(mp):
                qc["mask_missing"].append(dict(split=split, image_id=image_id, lesion_id=int(lid)))
                continue
            m = load_mask(mp)
            if tuple(m.shape) != img_hw:
                mismatches.append(dict(split=split, image_id=image_id, lesion_id=int(lid),
                                       mask_h=m.shape[0], mask_w=m.shape[1],
                                       image_h=img_hw[0], image_w=img_hw[1],
                                       mask_path=mp, image_path=ip))
                continue
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                qc["mask_empty"].append(dict(split=split, image_id=image_id, lesion_id=int(lid)))
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            n_cc = int(cv2.connectedComponents(m, connectivity=8)[0] - 1)
            fill = float(len(xs)) / float((x1 - x0) * (y1 - y0))
            lesions.append(dict(mask_path=mp, lesion_id=int(lid), label=lab, family=family, pathology=pathology,
                                bbox=[x0, y0, x1, y1], mask_shape=list(m.shape),
                                n_components=n_cc, fill_ratio=fill))
        if not lesions:
            reasons["nessuna maschera valida"] += 1
            continue
        if cfg.ltn_attrs:
            # facts of this lesion from the facts file, key (image_id, lesion_id of the manifest)
            for le in lesions:
                f = FACTS.get((image_id, le["lesion_id"]))
                if f is not None and f["family"] != family:
                    raise ValueError(f"{image_id} lesion {le['lesion_id']}: family {f['family']} in the facts file, "
                                     f"{family} in the image")
                le["in_facts_file"] = f is not None
                le["attrs"] = f["attrs"] if f is not None else None
        records.append(dict(image_id=image_id, base_key=info["base_key"],
                            patient=info["patient"], side=info["side"], view=info["view"],
                            image_path=ip, image_shape=list(img_hw), family=family,
                            lesions=lesions, patient_id=str(row.get("patient_id", "unknown")),
                            merged_from=None))
    n_les = sum(len(r["lesions"]) for r in records)
    print(f"  {split}: {len(records)} record / {n_les} lesioni da {len(split_df)} righe | "
          f"scartate: {dict(reasons)}")
    return records


MISMATCHES: List[dict] = []
QC = defaultdict(list)
t0 = time.time()
train_records = build_records(splits["train"], "train", MISMATCHES, QC)
val_records = build_records(splits["val"], "val", MISMATCHES, QC)
print(f"({time.time() - t0:.0f}s)")
if cfg.ltn_attrs:
    for _name, _recs in (("train", train_records), ("val", val_records)):
        _las = [le.get("attrs") for r in _recs for le in r["lesions"]]
        _nofile = sum(not le.get("in_facts_file", False) for r in _recs for le in r["lesions"])
        print(f"  fatti BI-RADS {_name}: {sum(a is not None for a in _las)} lesioni con fatti su {len(_las)} "
              f"({_nofile} assenti dal file dei fatti) | {LTNA.describe(_las)}")

for key, rows in QC.items():
    p = os.path.join(RUN_DIR, f"qc_{key}.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"  {key:<14} n={len(rows):>4} -> {p}")

if MISMATCHES:
    p = os.path.join(RUN_DIR, "qc_mask_image_shape_mismatch.csv")
    pd.DataFrame(MISMATCHES).to_csv(p, index=False)
    raise RuntimeError(
        f"{len(MISMATCHES)} maschere hanno dimensione diversa dalla loro immagine "
        f"(elenco in {p}). Il box di quelle lesioni non è nel sistema di coordinate "
        f"dell'immagine: il notebook si ferma qui finché non è risolto.")
print("\ncontrollo maschera ↔ immagine: tutte le maschere hanno la dimensione della loro immagine")

# ====================================================================================================
# 5. Duplicated mammograms (Mass + Calc of the same image)
# ====================================================================================================
def png_fingerprint(path: str) -> Tuple[tuple, str, str]:
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    return tuple(a.shape), str(a.dtype), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def merge_verified(records: List[dict], split: str, log: List[dict]) -> List[dict]:
    by_key = defaultdict(list)
    for r in records:
        by_key[r["base_key"]].append(r)
    out, n_merged, n_kept = [], 0, 0
    for key, group in by_key.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        fams = sorted(g["family"] for g in group)
        same_name = (len({(g["patient"], g["side"], g["view"]) for g in group}) == 1 and
                     len({g["patient_id"] for g in group}) == 1)
        prints = [png_fingerprint(g["image_path"]) for g in group]
        same_shape = len({p[0] for p in prints}) == 1
        same_dtype = len({p[1] for p in prints}) == 1
        same_hash = len({p[2] for p in prints}) == 1
        ok = same_name and same_shape and same_dtype and same_hash
        log.append(dict(split=split, base_key=key, records=[g["image_id"] for g in group],
                        families=fams, same_name=same_name, same_shape=same_shape,
                        same_dtype=same_dtype, same_pixels=same_hash, merged=ok))
        if not ok:
            out.extend(group)
            n_kept += 1
            continue
        base = dict(group[0])
        base["lesions"] = [le for g in group for le in g["lesions"]]
        base["family"] = "BOTH" if len(set(fams)) > 1 else fams[0]
        base["merged_from"] = [g["image_id"] for g in group]
        out.append(base)
        n_merged += 1
    print(f"  {split}: {n_merged} gruppi uniti, {n_kept} gruppi lasciati separati "
          f"-> {len(records)} record diventano {len(out)}")
    return out


DUP_LOG: List[dict] = []
train_records = merge_verified(train_records, "train", DUP_LOG)
val_records = merge_verified(val_records, "val", DUP_LOG)

dup_df = pd.DataFrame(DUP_LOG)
dup_df.to_csv(os.path.join(RUN_DIR, "qc_duplicates.csv"), index=False)
if len(dup_df):
    not_merged = dup_df[~dup_df["merged"]]
    not_merged.to_csv(os.path.join(RUN_DIR, "qc_duplicates_not_merged.csv"), index=False)
    print(f"gruppi doppi: {len(dup_df)} | uniti {int(dup_df['merged'].sum())} | "
          f"non uniti {len(not_merged)}")
    if len(not_merged):
        print(not_merged[["split", "base_key", "same_name", "same_shape", "same_pixels"]].to_string())

shared = {r["base_key"] for r in train_records} & {r["base_key"] for r in val_records}
if shared:
    raise RuntimeError(f"{len(shared)} mammografie compaiono sia in train sia in val: {sorted(shared)[:5]}")
print("nessuna mammografia condivisa tra train e val")

# ====================================================================================================
# 6. Dataset summary and mask diagnostics
# ====================================================================================================
def summary(records, name):
    c = Counter(CLASS_NAMES[le["label"]] for r in records for le in r["lesions"])
    tot = sum(c.values())
    print(f"{name:<6} immagini {len(records):>5} | lesioni {tot:>5} | " +
          "  ".join(f"{k}={c[k]} ({100 * c[k] / max(1, tot):.1f}%)" for k in list(CLASS_NAMES.values())[1:]))


summary(train_records, "train")
summary(val_records, "val")

rows = [dict(split=s, family=le["family"], n_components=le["n_components"],
             fill_ratio=le["fill_ratio"])
        for s, recs in (("train", train_records), ("val", val_records))
        for r in recs for le in r["lesions"]]
mdf = pd.DataFrame(rows)
print("\nmaschere: componenti connesse e riempimento del box (area maschera / area box)")
print(mdf.groupby(["split", "family"]).agg(
    n=("n_components", "size"),
    piu_di_1_componente=("n_components", lambda x: float((x > 1).mean())),
    componenti_mediana=("n_components", "median"),
    riempimento_mediano=("fill_ratio", "median"),
    riempimento_q1=("fill_ratio", lambda x: float(x.quantile(0.25)))).round(3).to_string())

# ====================================================================================================
# Raw intensity range of the PNGs (information only)
# ====================================================================================================
_rng = random.Random(cfg.seed)
_probe = _rng.sample(train_records, min(20, len(train_records)))
for r in _probe:
    a = cv2.imread(r["image_path"], cv2.IMREAD_UNCHANGED)
    print(f"{str(a.dtype):<7} min {int(a.min()):>6} max {int(a.max()):>6}  {r['image_id']}")

# ====================================================================================================
# 7. Dataset
# ====================================================================================================
import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")   # in-memory images are read-only: the code always copies them before changing them

IMAGE_CACHE = {}   # image_path -> (resized float32 image, maximum value of the PNG, original (h, w))


def apply_clahe(img: np.ndarray, full_scale: float) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) on the resized image.
    cv2 works on integers only: the float image goes back to the depth of its PNG (uint16 for 16 bit,
    uint8 for 8 bit), is equalised tile by tile, and comes back as float32 on the same scale [0, full_scale],
    so the normalisation in __getitem__ is unchanged. Deterministic: draws no random numbers.
    Applied after the resize: the 8x8 grid covers the same fraction of the breast at any resolution,
    and it costs ~1 ms per image instead of seconds at full resolution (deviation from Abdikenov 2025,
    who do not state at which resolution they apply it)."""
    if full_scale == 65535.0:
        dtype = np.uint16
    elif full_scale == 255.0:
        dtype = np.uint8
    else:
        raise ValueError(f"CLAHE: PNG con valore massimo {full_scale}, attesi 8 o 16 bit")
    q = np.clip(np.rint(img), 0, full_scale).astype(dtype)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    return clahe.apply(q).astype(np.float32)


def load_resized(r):
    """Read the PNG and resize: the same lines that the notebook has in LesionDataset.__getitem__.
    Nothing random: flip and normalisation stay in __getitem__, at every batch."""
    img = cv2.imread(r["image_path"], cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(r["image_path"])
    if img.ndim == 3:
        img = img[..., 0]
    full_scale = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    img = img.astype(np.float32)
    h, w = img.shape
    if [h, w] != list(r["image_shape"]):
        raise RuntimeError(f"{r['image_id']}: immagine {h}x{w}, registrata "
                           f"{r['image_shape']}: il file è cambiato dopo la costruzione dei record")

    scale = cfg.image_size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if cfg.clahe:                      # preprocessing: the same on train and validation, also in the cache
        img = apply_clahe(img, full_scale)
    img.setflags(write=False)          # protection: nobody can modify the image in memory
    return img, full_scale, (h, w)


# ====================================================================================================
# 7b. Geometric augmentation: scaling + rotation
# ====================================================================================================
# Idea: one affine transform (rotation + zoom about the centre), applied to the image AND to the lesion
# masks. The new box is taken from the transformed mask, not from the old box: rotating the 4 corners
# of a box enlarges it, while the mask gives the tight box around the lesion.

MASK_CACHE = {}   # (mask_path, nh, nw) -> resized uint8 mask, read-only


def resized_mask(path: str, nh: int, nw: int) -> np.ndarray:
    """Lesion mask at the same resolution as the image in the dataset (long side 512).
    Read from disk only once, then kept in memory."""
    key = (path, nh, nw)
    m = MASK_CACHE.get(key)
    if m is None:
        m = cv2.resize(load_mask(path), (nw, nh), interpolation=cv2.INTER_NEAREST)
        m.setflags(write=False)
        MASK_CACHE[key] = m
    return m


def affine_matrix(nh: int, nw: int, angle_deg: float, scale: float) -> np.ndarray:
    """2x3 matrix: rotation by angle_deg degrees and zoom by a factor scale, about the centre."""
    return cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle_deg, scale)


def warp_sample(img, boxes, masks, M):
    """Apply M to the image and the masks. Returns (image, boxes, masks), or None if a lesion
    would leave the image even partly, or would vanish."""
    nh, nw = img.shape
    for x0, y0, x1, y1 in boxes:                      # 1) no lesion cut by the border
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=np.float64)
        p = corners @ M.T                              # corners of the box after the transform
        if p[:, 0].min() < 0 or p[:, 1].min() < 0 or p[:, 0].max() > nw or p[:, 1].max() > nh:
            return None
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,          # 2) image: a new copy,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)      #    the cache stays untouched
    new_boxes, new_masks = [], []
    for m in masks:                                    # 3) new boxes from the transformed masks
        wm = cv2.warpAffine(m, M, (nw, nh), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ys, xs = np.nonzero(wm)
        if len(xs) == 0:                               # tiny lesion vanished: no augmentation
            return None
        new_boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        new_masks.append(wm)
    return out, new_boxes, new_masks


AUG_STATS = Counter()   # how many times the augmentation was applied / refused


def random_affine(img, boxes, masks, tries: int = 5):
    """Draw a random angle and scale; if the transform would cut a lesion, try again.
    After `tries` failed attempts, return the original image. Returns (image, boxes, masks)."""
    nh, nw = img.shape
    for _ in range(tries):
        angle = random.uniform(-cfg.rotate_deg, cfg.rotate_deg)
        scale = random.uniform(*cfg.scale_range)
        res = warp_sample(img, boxes, masks, affine_matrix(nh, nw, angle, scale))
        if res is not None:
            AUG_STATS["applied"] += 1
            return res
    AUG_STATS["refused"] += 1
    return img, boxes, masks


def random_photometric(img: np.ndarray, full_scale: float) -> np.ndarray:
    """Photometric jitter (SymBreastNet §4.2): contrast about the image mean, then gamma, on the [0, 1] scale.
    Returns a new float32 array on the original scale [0, full_scale]; the cached image is not touched.
    Contrast as torchvision adjust_contrast: x' = mean + c (x - mean), clipped to [0, 1]."""
    x = img / full_scale                               # new array in [0, 1]
    c = random.uniform(*cfg.photo_contrast)
    m = float(x.mean())
    x = np.clip(m + c * (x - m), 0.0, 1.0)
    g = random.choice(cfg.photo_gamma)
    x = np.power(x, g)
    AUG_STATS["photo"] += 1
    return (x * full_scale).astype(np.float32)


def _rng() -> np.random.Generator:
    """numpy generator seeded from the `random` stream: reproducible with cfg.seed, and called only when an
    augmentation is applied, so with every probability at 0 the stream is unchanged."""
    return np.random.default_rng(random.getrandbits(32))


def random_noise(img: np.ndarray, full_scale: float) -> np.ndarray:
    """Gaussian noise on the breast pixels only (the black background stays 0), on the [0, 1] scale.
    sigma is drawn per image in [0, noise_std]. Returns a new float32 array on the original scale."""
    x = img / full_scale
    sigma = random.uniform(0.0, cfg.noise_std)
    breast = x > 0
    x = np.where(breast, np.clip(x + _rng().normal(0.0, sigma, x.shape), 0.0, 1.0), 0.0)
    AUG_STATS["noise"] += 1
    return (x * full_scale).astype(np.float32)


def random_erase(img: np.ndarray, boxes, full_scale: float, tries: int = 10) -> np.ndarray:
    """Random erasing (Cutout): one rectangle filled with the mean breast intensity. The rectangle must have its
    centre on the breast and must not touch any lesion box, so the labels stay true.
    After `tries` failed draws the image is returned unchanged."""
    nh, nw = img.shape
    on_breast = img > 0.05 * full_scale
    if not on_breast.any():
        AUG_STATS["erase_refused"] += 1
        return img
    fill = float(img[on_breast].mean())
    for _ in range(tries):
        area = random.uniform(*cfg.erase_area) * nh * nw
        ratio = math.exp(random.uniform(math.log(0.3), math.log(3.3)))    # height / width
        eh, ew = int(round(math.sqrt(area * ratio))), int(round(math.sqrt(area / ratio)))
        if eh < 1 or ew < 1 or eh >= nh or ew >= nw:
            continue
        y0, x0 = random.randint(0, nh - eh), random.randint(0, nw - ew)
        y1, x1 = y0 + eh, x0 + ew
        if not on_breast[(y0 + y1) // 2, (x0 + x1) // 2]:
            continue
        if any(x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1 for b in boxes):
            continue
        out = img.astype(np.float32, copy=True)        # the cached image is read-only
        out[y0:y1, x0:x1] = fill
        AUG_STATS["erase"] += 1
        return out
    AUG_STATS["erase_refused"] += 1
    return img


def random_elastic(img: np.ndarray, boxes, masks, tries: int = 3):
    """Smooth elastic deformation: random displacements on a coarse elastic_grid x elastic_grid grid
    (at most elastic_px pixels), upsampled with cubic interpolation, applied with cv2.remap to the image and to
    the lesion masks. New boxes from the deformed masks, as in 7b. Returns (image, boxes, masks)."""
    nh, nw = img.shape
    g = cfg.elastic_grid
    xx, yy = np.meshgrid(np.arange(nw, dtype=np.float32), np.arange(nh, dtype=np.float32))
    for _ in range(tries):
        rng = _rng()
        dx = cv2.resize(rng.uniform(-cfg.elastic_px, cfg.elastic_px, (g, g)).astype(np.float32), (nw, nh),
                        interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(rng.uniform(-cfg.elastic_px, cfg.elastic_px, (g, g)).astype(np.float32), (nw, nh),
                        interpolation=cv2.INTER_CUBIC)
        mx, my = xx + dx, yy + dy                      # output(x, y) = input(mx, my)
        new_boxes, new_masks = [], []
        for m in masks:
            wm = cv2.remap(m, mx, my, interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ys, xs = np.nonzero(wm)
            if len(xs) == 0:                           # lesion vanished: draw again
                break
            new_boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
            new_masks.append(wm)
        else:
            out = cv2.remap(np.ascontiguousarray(img), mx, my, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            AUG_STATS["elastic"] += 1
            return out, new_boxes, new_masks
    AUG_STATS["elastic_refused"] += 1
    return img, boxes, masks


class LesionDataset(Dataset):
    def __init__(self, records: List[dict], train: bool):
        self.records = records
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        cached = IMAGE_CACHE.get(r["image_path"])
        img, full_scale, (h, w) = cached if cached is not None else load_resized(r)

        scale = cfg.image_size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

        boxes, labels, mask_paths, lesion_attrs = [], [], [], []
        for le in r["lesions"]:
            x0, y0, x1, y1 = [c * scale for c in le["bbox"]]
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1, y1])
            labels.append(int(le["label"]))
            mask_paths.append(le["mask_path"])
            lesion_attrs.append(le.get("attrs"))     # same order as boxes; no augmentation reorders or drops boxes

        # scaling + rotation (7b). "affine_p > 0" before random.random(): with affine_p = 0 the generator
        # is not touched and the flip sequence stays identical to gdn4iavp
        masks = None
        if self.train and cfg.affine_p > 0 and random.random() < cfg.affine_p:
            masks = [resized_mask(p, nh, nw) for p in mask_paths]
            img, boxes, masks = random_affine(img, boxes, masks)

        # elastic deformation: after the affine step, on the masks it has already transformed
        if self.train and cfg.elastic_p > 0 and random.random() < cfg.elastic_p:
            if masks is None:
                masks = [resized_mask(p, nh, nw) for p in mask_paths]
            img, boxes, masks = random_elastic(img, boxes, masks)

        if self.train and random.random() < cfg.hflip_p:
            img = np.ascontiguousarray(img[:, ::-1])
            boxes = [[nw - b[2], b[1], nw - b[0], b[3]] for b in boxes]

        if self.train and cfg.vflip_p > 0 and random.random() < cfg.vflip_p:
            img = np.ascontiguousarray(img[::-1, :])
            boxes = [[b[0], nh - b[3], b[2], nh - b[1]] for b in boxes]
            AUG_STATS["vflip"] += 1

        # photometric jitter: after the flip, so with photo_p = 0 the random stream is the same as before
        if self.train and cfg.photo_p > 0 and random.random() < cfg.photo_p:
            img = random_photometric(img, full_scale)

        if self.train and cfg.noise_p > 0 and random.random() < cfg.noise_p:
            img = random_noise(img, full_scale)

        if self.train and cfg.erase_p > 0 and random.random() < cfg.erase_p:
            img = random_erase(img, boxes, full_scale)

        sample = torch.from_numpy(img)[None].repeat(3, 1, 1)
        if cfg.normalization == "zscore":
            sample = sample - np.mean(sample.numpy())
            std = np.std(sample.numpy())
            if not np.isfinite(std) or std == 0:
                raise ValueError(f"{r['image_id']}: deviazione standard {std}, z-score impossibile")
            sample = sample / std
        elif cfg.normalization == "imagenet":
            # [0, 1] according to the PNG depth (65535 for 16 bit); ImageNet mean and std in build_model()
            sample = sample / full_scale
        else:
            raise ValueError(f"normalization {cfg.normalization!r}: usa 'imagenet' o 'zscore'")

        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([idx])}
        if cfg.ltn_attrs:
            if len(target["boxes"]) != len(lesion_attrs):
                raise RuntimeError(f"{r['image_id']}: {len(target['boxes'])} box ma {len(lesion_attrs)} attributi")
            # LTN: attr_<head> [n, K] and attr_<head>_valid [n] per lesion, in the order of the boxes
            for k, v in LTNA.target_tensors(lesion_attrs).items():
                target[k] = torch.from_numpy(v)
        return sample, target, r["image_id"]


def collate(batch):
    imgs, targets, ids = zip(*batch)
    return list(imgs), list(targets), list(ids)


def make_loader(records, train: bool, batch_size: Optional[int] = None) -> DataLoader:
    return DataLoader(LesionDataset(records, train=train),
                      batch_size=batch_size or (cfg.batch_size if train else 1),
                      shuffle=train, num_workers=cfg.num_workers, collate_fn=collate,
                      pin_memory=(DEVICE.type == "cuda"), drop_last=False)


if cfg.cache_images:
    from concurrent.futures import ThreadPoolExecutor
    _t0 = time.time()
    _todo = {r["image_path"]: r for r in train_records + val_records}
    with ThreadPoolExecutor(max(1, cfg.cache_threads)) as _ex:
        for _path, _val in zip(_todo, _ex.map(load_resized, _todo.values())):
            IMAGE_CACHE[_path] = _val
    print(f"immagini in memoria: {len(IMAGE_CACHE)} (train + val), "
          f"{sum(v[0].nbytes for v in IMAGE_CACHE.values()) / 1e9:.2f} GB, {time.time() - _t0:.0f}s")
else:
    print("immagini lette da disco a ogni batch (cache_images=False)")

# ====================================================================================================
# Dataset checks
# ====================================================================================================
def selftest_dataset(records, n=8):
    ds = LesionDataset(records, train=False)
    for i in range(min(n, len(ds))):
        img, t, iid = ds[i]
        assert img.shape[0] == 3 and img.dtype == torch.float32, iid
        assert max(img.shape[1:]) == cfg.image_size, f"{iid}: lato lungo {max(img.shape[1:])}"
        m, s = float(img.mean()), float(img.std(unbiased=False))
        if cfg.normalization == "zscore":
            assert abs(m) < 1e-3 and abs(s - 1) < 1e-3, f"{iid}: z-score non riuscita (mean {m}, std {s})"
        else:
            assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0 + 1e-6, f"{iid}: valori fuori da [0, 1]"
        b = t["boxes"]
        assert len(b) == len(t["labels"]) > 0, iid
        assert torch.all(b[:, 2] > b[:, 0]) and torch.all(b[:, 3] > b[:, 1]), iid
        assert torch.all(b[:, :2] >= -1) and torch.all(b[:, 2] <= img.shape[2] + 1) \
            and torch.all(b[:, 3] <= img.shape[1] + 1), f"{iid}: box fuori immagine"
    print(f"  dataset ok su {min(n, len(ds))} campioni (normalizzazione: {cfg.normalization})")


def selftest_box_vs_resized_mask(records, n=20):
    """The scaled box must coincide with the box of the resized mask."""
    rng = random.Random(0)
    worst = 1.0
    for r in rng.sample(records, min(n, len(records))):
        h, w = r["image_shape"]
        scale = cfg.image_size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        for le in r["lesions"]:
            m = cv2.resize(load_mask(le["mask_path"]), (nw, nh), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            bm = torch.tensor([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=torch.float32)
            bs = torch.tensor([[c * scale for c in le["bbox"]]], dtype=torch.float32)
            worst = min(worst, float(box_iou(bm, bs)))
    assert worst > 0.7, f"box scalato e box della maschera ridimensionata divergono (IoU min {worst:.3f})"
    print(f"  box ↔ maschera ridimensionata: IoU minima {worst:.3f}")


selftest_dataset(train_records)
selftest_dataset(val_records)
selftest_box_vs_resized_mask(val_records)

# ====================================================================================================
# 8. Examples: plots not drawn; same consumption of the random generator as the notebook
# ====================================================================================================
# At this point the notebook shows n_examples training images with a random flip (section 8a): each image
# consumes one number of the global `random` generator. The script draws no plots but consumes the same numbers,
# so the flip sequence in training is the same as in the notebook. Section 8b uses a local generator: no effect.
for _ in range(min(cfg.n_examples, len(train_records))):
    random.random()

# ====================================================================================================
# 9. Model
# ====================================================================================================
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def input_norm():
    """mean and std applied by GeneralizedRCNNTransform"""
    if cfg.normalization == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]       # z-score already done in the Dataset


def build_model() -> torch.nn.Module:
    w = "DEFAULT" if cfg.pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=w, weights_backbone=w,
        trainable_backbone_layers=cfg.trainable_backbone_layers,
        min_size=cfg.image_size, max_size=cfg.image_size,
        image_mean=input_norm()[0], image_std=input_norm()[1])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)


    # RPN anchors and thresholds from Config (notebook 09)
    assert len(cfg.anchor_sizes) == 5, f"servono 5 livelli FPN (P2..P6), dati {len(cfg.anchor_sizes)}"
    assert 0 < cfg.rpn_bg_iou <= cfg.rpn_fg_iou <= 1, (cfg.rpn_bg_iou, cfg.rpn_fg_iou)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple(tuple(float(s) for s in level) for level in cfg.anchor_sizes),
        aspect_ratios=(tuple(float(r) for r in cfg.anchor_ratios),) * len(cfg.anchor_sizes))
    n_per_loc = model.rpn.anchor_generator.num_anchors_per_location()
    assert len(set(n_per_loc)) == 1, f"stesso numero di dimensioni su ogni livello: {n_per_loc}"
    if n_per_loc[0] != model.rpn.head.cls_logits.out_channels:
        # anchors per location different from COCO's (3): new cls_logits and bbox_pred,
        # the 3x3 convolution of the RPN head keeps the COCO weights
        new_head = RPNHead(model.backbone.out_channels, n_per_loc[0])
        new_head.conv.load_state_dict(model.rpn.head.conv.state_dict())
        model.rpn.head = new_head
    # anchors positive if IoU >= rpn_fg_iou, background if IoU < rpn_bg_iou, ignored in between
    model.rpn.proposal_matcher = Matcher(cfg.rpn_fg_iou, cfg.rpn_bg_iou, allow_low_quality_matches=True)

    # LTN (thesis Ch. 6): same RoI modules and weights, plus attribute heads and the knowledge-base losses
    if cfg.ltn:
        # replaces model.roi_heads with LTNRoIHeads (same modules and weights) + concept heads and LTN losses
        LTNH.install(model, class_loss=cfg.ltn_class_loss, class_agg=cfg.ltn_class_agg,
                     class_bg_weight=cfg.ltn_class_bg_w, attr_loss=cfg.ltn_attr_loss,
                     axioms=cfg.ltn_axioms, ax_guard=cfg.ltn_ax_guard, ax_family=cfg.ltn_ax_family,
                     aux=tuple(cfg.ltn_aux), ent_target=cfg.ltn_ent_target,
                     weights={"class": cfg.ltn_w_class, "find": cfg.ltn_w_find, "attr": cfg.ltn_w_attr,
                              "clin": cfg.ltn_w_clin, "ent": cfg.ltn_w_ent, "kl": cfg.ltn_w_kl, "rec": cfg.ltn_w_rec},
                     norm=cfg.ltn_norm)
        rh = model.roi_heads
        print(f"LTN v{LTNH.VERSION}     {rh.describe()}")
        print(f"LTN        p={'schedule 2 -> 10' if cfg.ltn_p <= 0 else cfg.ltn_p}")
        # every run log starts with the axioms it trains, in the paper's notation
        for _a in rh.axioms:
            print("LTN        " + LTNL.axiom_text(_a))

    return model


model = build_model().to(DEVICE)

assert list(model.transform.image_mean) == input_norm()[0], model.transform.image_mean
assert list(model.transform.image_std) == input_norm()[1], model.transform.image_std
assert model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f"parametri addestrabili {n_tr / 1e6:.3f} M su {n_all / 1e6:.3f} M")
print(f"transform  min_size={model.transform.min_size} max_size={model.transform.max_size} "
      f"mean={model.transform.image_mean} std={model.transform.image_std}")
print(f"anchor     {model.rpn.anchor_generator.sizes}  ratios {model.rpn.anchor_generator.aspect_ratios[0]}  "
      f"| {model.rpn.anchor_generator.num_anchors_per_location()[0]} per posizione")
print(f"RPN        positivo >= {model.rpn.proposal_matcher.high_threshold}  "
      f"negativo < {model.rpn.proposal_matcher.low_threshold}  "
      f"campioni {model.rpn.fg_bg_sampler.batch_size_per_image} / {model.rpn.fg_bg_sampler.positive_fraction}")
print(f"RoI        fg >= {model.roi_heads.proposal_matcher.high_threshold}  "
      f"campioni {model.roi_heads.fg_bg_sampler.batch_size_per_image} / {model.roi_heads.fg_bg_sampler.positive_fraction}")
print(f"postproc   score >= {model.roi_heads.score_thresh}  NMS {model.roi_heads.nms_thresh}  "
      f"max {model.roi_heads.detections_per_img} per immagine")

# ====================================================================================================
# 10. Coverage of the model's **real** anchors
# ====================================================================================================
@torch.no_grad()
def anchor_coverage(model, records) -> pd.DataFrame:
    model.eval()
    hi = model.rpn.proposal_matcher.high_threshold
    lo = model.rpn.proposal_matcher.low_threshold
    ds = LesionDataset(records, train=False)
    rows = []
    for i in range(len(ds)):
        img, t, iid = ds[i]
        images, targets = model.transform([img.to(DEVICE)], [{k: v.to(DEVICE) for k, v in t.items()}])
        feats = model.backbone(images.tensors)
        anchors = model.rpn.anchor_generator(images, list(feats.values()))[0]
        gt = targets[0]["boxes"]
        iou = box_iou(gt, anchors)
        best = iou.max(dim=1).values
        npos = (iou >= hi).sum(dim=1)
        for b, l, bi, n in zip(gt.cpu(), targets[0]["labels"].cpu(), best.cpu(), npos.cpu()):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            rows.append(dict(family=LABEL2FAMILY[int(l)], longest_side=max(w, h),
                             best_iou=float(bi), n_pos_anchors=int(n)))
    df = pd.DataFrame(rows)
    print(f"soglie RPN lette dal modello: positivo >= {hi}, negativo < {lo}")
    print(df.groupby("family").agg(
        n=("best_iou", "size"),
        lato_mediano_px=("longest_side", "median"),
        iou_migliore_mediana=("best_iou", "median"),
        quota_con_anchor_positivo=("best_iou", lambda x: float((x >= hi).mean())),
        quota_sotto_negativo=("best_iou", lambda x: float((x < lo).mean())),
        anchor_positivi_mediana=("n_pos_anchors", "median")).round(3).to_string())
    return df


cov_df = anchor_coverage(model, val_records)
cov_df.to_csv(os.path.join(RUN_DIR, "anchor_coverage_val.csv"), index=False)

# ====================================================================================================
# 11. Metrics
# ====================================================================================================
def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.clip(aa[:, None] + ab[None, :] - inter, 1e-9, None)).astype(np.float32)


def froc_curve(preds, gts, family: str, iou_thr: float, fp_points: tuple,
               ignore_duplicates: bool) -> Tuple[float, Dict[float, float]]:
    fam_ids = [i for i, f in LABEL2FAMILY.items() if f == family]
    n_images = len(gts)
    rows, n_gt = [], 0
    for p, g in zip(preds, gts):
        gb = g["boxes"][np.isin(g["labels"], fam_ids)]
        n_gt += len(gb)
        pm = np.isin(p["labels"], fam_ids)
        pb, ps = p["boxes"][pm], p["scores"][pm]
        if len(pb) == 0:
            continue
        iou = box_iou_np(pb, gb)
        taken = set()
        for k in np.argsort(-ps):
            hit, dup = -1, False
            for c in (np.argsort(-iou[k]) if iou.shape[1] else []):
                if iou[k, c] < iou_thr:
                    break
                if c not in taken:
                    hit = c
                    break
                dup = True
            if hit >= 0:
                taken.add(hit)
                rows.append((float(ps[k]), 1))
            elif dup and ignore_duplicates:
                continue
            else:
                rows.append((float(ps[k]), 0))
    if n_gt == 0:
        raise ValueError(f"nessuna lesione {family}")
    if not rows:
        return 0.0, {fp: 0.0 for fp in fp_points}
    rows.sort(key=lambda t: -t[0])
    tp = np.cumsum([r[1] for r in rows]); fp = np.cumsum([1 - r[1] for r in rows])
    sens, fppi = tp / n_gt, fp / n_images
    at = {pt: (float(sens[fppi <= pt].max()) if (fppi <= pt).any() else 0.0) for pt in fp_points}
    return float(np.mean(list(at.values()))), at


def image_malignancy_auc(preds, gts) -> float:
    y, s = [], []
    for p, g in zip(preds, gts):
        y.append(int(np.isin(g["labels"], list(MALIGNANT_IDS)).any()))
        m = np.isin(p["labels"], list(MALIGNANT_IDS))
        s.append(float(p["scores"][m].max()) if m.any() else 0.0)
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, gts, ids = [], [], []
    for imgs, targets, image_ids in tqdm(loader, desc="valutazione", leave=False):
        imgs = [i.to(DEVICE, non_blocking=True) for i in imgs]
        with torch.autocast(device_type=DEVICE.type, enabled=(cfg.amp and DEVICE.type == "cuda")):
            out = model(imgs)
        for o, t, iid in zip(out, targets, image_ids):
            preds.append(dict(boxes=o["boxes"].float().cpu().numpy(),
                              scores=o["scores"].float().cpu().numpy(),
                              labels=o["labels"].cpu().numpy()))
            gts.append(dict(boxes=t["boxes"].numpy(), labels=t["labels"].numpy()))
            ids.append(iid)
    return preds, gts, ids


def evaluate(model, loader) -> dict:
    preds, gts, _ = predict(model, loader)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.update([dict(boxes=torch.from_numpy(p["boxes"]), scores=torch.from_numpy(p["scores"]),
                        labels=torch.from_numpy(p["labels"])) for p in preds],
                  [dict(boxes=torch.from_numpy(g["boxes"]), labels=torch.from_numpy(g["labels"]))
                   for g in gts])
    m = metric.compute()
    out = dict(map=float(m["map"]), map_50=float(m["map_50"]), map_75=float(m["map_75"]),
               n_detections=int(sum(len(p["boxes"]) for p in preds)))
    classes = m["classes"].tolist() if m["classes"].ndim else [int(m["classes"])]
    per_cls = m["map_per_class"].tolist() if m["map_per_class"].ndim else [float(m["map_per_class"])]
    for cid, ap in zip(classes, per_cls):
        out[f"ap_{CLASS_NAMES[int(cid)]}"] = float(ap)
    for fam in ("MASS", "CALC"):
        for dedup, suffix in ((False, ""), (True, "_dedup")):
            mean_s, at = froc_curve(preds, gts, fam, cfg.froc_iou, cfg.froc_fp_points, dedup)
            out[f"froc_{fam.lower()}_mean{suffix}"] = mean_s
            for k, v in at.items():
                out[f"froc_{fam.lower()}{suffix}@{k}"] = v
    out["image_auc_malignancy"] = image_malignancy_auc(preds, gts)
    return out

# ====================================================================================================
# 12. Validation loss — without touching the randomness of training
# ====================================================================================================
from tqdm.auto import tqdm


def _rng_state():
    return dict(torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                python=random.getstate(), numpy=np.random.get_state())


def _set_rng_state(s):
    # torch.load(LAST_PT, map_location=DEVICE) moves EVERY tensor of the checkpoint to the GPU, the random
    # states included, but torch accepts them only as CPU ByteTensors: hence .cpu()
    torch.set_rng_state(s["torch_cpu"].cpu())
    if s["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([t.cpu() for t in s["torch_cuda"]])
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])


def validation_losses(model, loader, seed: int) -> Dict[str, float]:
    if not cfg.pretrained:
        raise RuntimeError("validation_losses richiede pretrained=True (batch norm congelata)")
    saved = _rng_state()
    was_training = model.training
    try:
        torch.manual_seed(seed)
        model.train()
        sums, nb, ltn_n = defaultdict(float), 0, defaultdict(int)
        with torch.no_grad():
            for imgs, targets, _ in tqdm(loader, desc="loss validation", leave=False):
                imgs = [i.to(DEVICE, non_blocking=True) for i in imgs]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                losses = model(imgs, targets)
                total = sum(losses.values())
                if not torch.isfinite(total):
                    raise FloatingPointError(f"loss di validazione non finita: "
                                             f"{ {k: float(v) for k, v in losses.items()} }")
                for k, v in losses.items():
                    sums[k] += float(v)
                sums["loss"] += float(total)
                # LTN: Sat / CE values of this batch (only the terms present in it)
                if cfg.ltn:
                    for k, v in model.roi_heads.last_log.items():
                        sums[k] += v
                        ltn_n[k] += 1
                nb += 1
    finally:
        model.train(was_training)
        _set_rng_state(saved)
    # losses: mean per batch (a batch without the term adds 0); LTN log values (Sat per axiom / head, CE per head):
    # mean over the batches where they exist, so an axiom whose domain is absent from a batch is not counted as 0
    return {k: v / max(1, ltn_n[k] if k in ltn_n else nb) for k, v in sums.items()}


# check: the function must not change the state of the generator used by training
_before = torch.get_rng_state().clone()
_ = validation_losses(model, make_loader(val_records[:2], train=False), seed=cfg.seed)
assert torch.equal(_before, torch.get_rng_state()), "validation_losses ha alterato il generatore CPU"
print("validation_losses lascia invariato lo stato dei generatori casuali")

# ====================================================================================================
# Run manifest (run_manifest.json): configuration, environment, data, model, optimiser, results, files
# ====================================================================================================
import platform

RUN_MANIFEST_JSON = os.path.join(RUN_DIR, "run_manifest.json")
REPORT_JSON = os.path.join(RUN_DIR, "report_validation.json")


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _file_info(path):
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return dict(path=path, bytes=st.st_size, sha256=_sha256(path),
                modified=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))


def _environment():
    env = dict(python=sys.version.split()[0], executable=sys.executable, platform=platform.platform(),
               torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
               opencv=cv2.__version__, pandas=pd.__version__, device=str(DEVICE))
    try:
        import torchmetrics
        env["torchmetrics"] = torchmetrics.__version__
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        env.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
    return env


def _split_summary(df, recs):
    per_class = Counter(CLASS_NAMES[le["label"]] for r in recs for le in r["lesions"])
    return dict(rows_in_split=len(df), images=len(recs), lesions=sum(per_class.values()), lesions_per_class=dict(per_class),
                merged_mammograms=sum(1 for r in recs if r["merged_from"]))


def _data_summary():
    return dict(input_files={p: _file_info(p) for p in (SPLITS_XLSX, MANIFEST_CSV, *FACTS_FILES)},
                images_dir=IMAGES_DIR, lesion_masks_dir=LESION_MASK_DIR,
                train=_split_summary(splits["train"], train_records), val=_split_summary(splits["val"], val_records),
                qc_lesions_discarded={k: len(v) for k, v in QC.items()},
                duplicate_groups=dict(total=len(dup_df), merged=int(dup_df["merged"].sum()) if len(dup_df) else 0),
                test_loaded=False)


def _model_summary(model):
    rpn, roi, tr = model.rpn, model.roi_heads, model.transform
    return dict(architecture="fasterrcnn_resnet50_fpn",
                pretrained_weights="COCO (torchvision DEFAULT)" if cfg.pretrained else None,
                num_classes=NUM_CLASSES, class_names=CLASS_NAMES,
                anchor_sizes=[list(s) for s in rpn.anchor_generator.sizes],
                anchor_aspect_ratios=list(rpn.anchor_generator.aspect_ratios[0]),
                anchors_per_location=rpn.anchor_generator.num_anchors_per_location()[0],
                rpn=dict(fg_iou=rpn.proposal_matcher.high_threshold, bg_iou=rpn.proposal_matcher.low_threshold,
                         batch_per_image=rpn.fg_bg_sampler.batch_size_per_image,
                         positive_fraction=rpn.fg_bg_sampler.positive_fraction),
                roi=dict(fg_iou=roi.proposal_matcher.high_threshold, bg_iou=roi.proposal_matcher.low_threshold,
                         batch_per_image=roi.fg_bg_sampler.batch_size_per_image,
                         positive_fraction=roi.fg_bg_sampler.positive_fraction,
                         score_thresh=roi.score_thresh, nms_thresh=roi.nms_thresh,
                         detections_per_img=roi.detections_per_img),
                transform=dict(min_size=list(tr.min_size), max_size=tr.max_size,
                               image_mean=list(tr.image_mean), image_std=list(tr.image_std)),
                normalization=cfg.normalization,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()))


def _optimizer_summary(optimiser, scaler):
    groups = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items() if k != "params"}
              for g in optimiser.param_groups]
    return dict(type=type(optimiser).__name__, param_groups=groups,
                parameters_optimized=sum(p.numel() for g in optimiser.param_groups for p in g["params"]),
                grad_scaler=dict(enabled=scaler.is_enabled(),
                                 scale=float(scaler.get_scale()) if scaler.is_enabled() else None),
                state_saved_in="last.pt: chiavi 'optimizer' (stato completo), 'scaler', 'rng'")


def _results():
    hist = []
    if os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            hist = json.load(f)
    scal = lambda r: {k: v for k, v in r.items() if isinstance(v, (int, float))}
    out = dict(epochs_completed=len(hist), epochs_planned=cfg.epochs)
    if hist:
        out["last_epoch"] = scal(hist[-1])
        ev = [h for h in hist if "val/map_50" in h]
        if ev:
            b = max(ev, key=lambda h: h["val/map_50"])
            out["best_epoch_by_map_50"] = scal(b)
    if os.path.exists(REPORT_JSON):
        with open(REPORT_JSON, "r", encoding="utf-8") as f:
            out["final_report"] = json.load(f).get("report")
    return out


if cfg.resume and os.path.exists(RUN_MANIFEST_JSON):
    with open(RUN_MANIFEST_JSON, "r", encoding="utf-8") as f:
        RUN_MANIFEST = json.load(f)
    RUN_MANIFEST.setdefault("resumes", []).append(dict(at=datetime.datetime.now().isoformat(timespec="seconds"),
                                                      command=" ".join(sys.argv), config=asdict(cfg)))
else:
    RUN_MANIFEST = dict(schema=1, run_name=cfg.run_name, run_dir=RUN_DIR,
                        started=datetime.datetime.now().isoformat(timespec="seconds"),
                        command=" ".join(sys.argv), resumes=[])
RUN_MANIFEST.update(config=asdict(cfg), environment=_environment(), data=_data_summary(), model=_model_summary(model))


def write_run_manifest(status, model=None, optimiser=None, scaler=None, wb=None):
    """Update run_manifest.json (atomic write). --config <this file> relaunches the same configuration."""
    RUN_MANIFEST["status"] = status
    RUN_MANIFEST["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    if model is not None:
        RUN_MANIFEST["model"] = _model_summary(model)
    if optimiser is not None:
        RUN_MANIFEST["optimizer"] = _optimizer_summary(optimiser, scaler)
    if wb is not None and getattr(wb, "run", None) is not None:
        RUN_MANIFEST["wandb"] = dict(project=cfg.wandb_project, entity=cfg.wandb_entity,
                                     id=wb.run.id, url=getattr(wb.run, "url", None))
    RUN_MANIFEST["results"] = _results()
    RUN_MANIFEST["files"] = {name: _file_info(p) for name, p in
                             (("best.pt", BEST_PT), ("last.pt", LAST_PT), ("history.json", HISTORY_JSON),
                              ("report_validation.json", REPORT_JSON))}
    RUN_MANIFEST["how_to"] = dict(
        rerun_same_config=f"python train_frcnn.py --config \"{RUN_MANIFEST_JSON}\" --run_name <nome nuovo>",
        resume=f"python train_frcnn.py --config \"{RUN_MANIFEST_JSON}\" --resume true",
        load_optimizer="ck = torch.load('last.pt', weights_only=False); optimiser.load_state_dict(ck['optimizer'])")
    tmp = RUN_MANIFEST_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(RUN_MANIFEST, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, RUN_MANIFEST_JSON)


write_run_manifest("preparazione completata (dati e modello pronti)", model=model)
print("manifest della run:", RUN_MANIFEST_JSON)

# ====================================================================================================
# 13. Training
# ====================================================================================================
from tqdm.auto import tqdm


class _WB:
    """Optional W&B: a network error must not stop training."""
    def __init__(self):
        self.run = None
        if not cfg.use_wandb:
            return
        try:
            import wandb
            self.wandb = wandb
            # --resume: continue the SAME W&B run (its id is in run_manifest.json), so the curves stay in one run
            prev_id = RUN_MANIFEST.get("wandb", {}).get("id") if cfg.resume else None
            if prev_id:
                self.run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                                      name=cfg.run_name, id=prev_id, resume="allow")
                self.run.config.update(asdict(cfg), allow_val_change=True)   # e.g. patience 25 -> 0
                print(f"  [wandb] ripresa del run {prev_id}")
            else:
                self.run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                                      name=cfg.run_name, config=asdict(cfg))
            print("  [wandb]", self.run.url)
        except Exception as e:  # noqa: BLE001
            print(f"  [wandb] disattivato: {type(e).__name__}: {e}")
            self.run = None
            return
        self._upload_script()

    def _upload_script(self):
        """Upload the script used for this training to W&B: Code tab (code artifact) and Files tab.
        The SHA-256 also goes into the summary, to compare it with run_manifest.json. An error does not stop training."""
        path = os.path.abspath(__file__)
        try:
            with open(path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            self.run.log_code(root=os.path.dirname(path),
                              include_fn=lambda p, *_: os.path.abspath(p) == path)
            self.wandb.save(path, base_path=os.path.dirname(path), policy="now")
            self.run.summary["script_name"] = os.path.basename(path)
            self.run.summary["script_sha256"] = sha
            print(f"  [wandb] script caricato: {os.path.basename(path)} (sha256 {sha[:12]}…)")
        except Exception as e:  # noqa: BLE001
            print(f"  [wandb] script non caricato: {type(e).__name__}: {e}")

    def log(self, d, step):
        if self.run is not None:
            try:
                self.wandb.log(d, step=step)
            except Exception as e:  # noqa: BLE001
                print(f"  [wandb] disattivato: {e}")
                self.run = None

    def finish(self):
        if self.run is not None:
            try:
                self.wandb.finish()
            except Exception:  # noqa: BLE001
                pass


def append_history(row: dict) -> None:
    hist = []
    if os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            hist = json.load(f)
    hist = [h for h in hist if h["epoch"] != row["epoch"]] + [row]
    hist.sort(key=lambda h: h["epoch"])
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(hist, f)


def save_checkpoint(path, model, epoch, metrics, optimiser=None, scaler=None):
    """best.pt: the model only. last.pt: also optimiser, scaler and random-generator state (for --resume)."""
    extra = {}
    if optimiser is not None:
        extra = dict(optimizer=optimiser.state_dict(), scaler=scaler.state_dict() if scaler is not None else None,
                     rng=_rng_state())
    if optimiser is not None:
        extra["manifest"] = json.loads(json.dumps(RUN_MANIFEST, default=str))   # copy of the manifest inside last.pt
    torch.save(dict(model=model.state_dict(), epoch=epoch, metrics=metrics, config=asdict(cfg),
                    class_names=CLASS_NAMES,
                    saved_at=datetime.datetime.now().isoformat(timespec="seconds"), **extra), path)


def train(model) -> dict:
    train_loader = make_loader(train_records, train=True)
    val_loader = make_loader(val_records, train=False)
    val_loss_loader = make_loader(val_records, train=False, batch_size=cfg.batch_size)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adam":
        optimiser = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimiser = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"optimizer {cfg.optimizer!r}: usa 'adam' o 'sgd'")
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and DEVICE.type == "cuda"))
    wb = _WB()
    write_run_manifest("in training", model=model, optimiser=optimiser, scaler=scaler, wb=wb)

    if cfg.lr_schedule not in ("constant", "step"):
        raise ValueError(f"lr_schedule {cfg.lr_schedule!r}: usa 'constant' o 'step'")
    if cfg.lr_schedule == "step" and cfg.lr_step_size < 1:
        raise ValueError(f"lr_step_size {cfg.lr_step_size}: deve essere >= 1")
    if cfg.patience < 0:
        raise ValueError(f"patience {cfg.patience}: deve essere >= 0 (0 = nessuno stop anticipato)")
    if cfg.photo_p > 0 and not (0 < cfg.photo_contrast[0] <= cfg.photo_contrast[1]
                                and len(cfg.photo_gamma) > 0 and min(cfg.photo_gamma) > 0):
        raise ValueError(f"fotometria: photo_contrast {cfg.photo_contrast} deve essere (basso, alto) > 0, "
                         f"photo_gamma {cfg.photo_gamma} valori > 0")
    for name in ("vflip_p", "noise_p", "erase_p", "elastic_p"):
        if not 0.0 <= getattr(cfg, name) <= 1.0:
            raise ValueError(f"{name} {getattr(cfg, name)}: deve essere in [0, 1]")
    if cfg.noise_p > 0 and not 0 < cfg.noise_std <= 0.2:
        raise ValueError(f"noise_std {cfg.noise_std}: deve essere in (0, 0.2] (scala [0, 1])")
    if cfg.erase_p > 0 and not 0 < cfg.erase_area[0] <= cfg.erase_area[1] < 0.5:
        raise ValueError(f"erase_area {cfg.erase_area}: (basso, alto) con 0 < basso <= alto < 0.5")
    if cfg.elastic_p > 0 and (cfg.elastic_grid < 2 or cfg.elastic_px <= 0):
        raise ValueError(f"elastic: elastic_grid {cfg.elastic_grid} >= 2, elastic_px {cfg.elastic_px} > 0")
    if not cfg.ltn and (cfg.ltn_attr_loss != "none" or cfg.ltn_axioms or cfg.ltn_aux or cfg.ltn_class_loss != "ce"):
        raise ValueError("ltn_class_loss / ltn_attr_loss / ltn_axioms / ltn_aux richiedono --ltn true")
    if cfg.ltn_part or cfg.ltn_w_part != 1.0:
        raise ValueError("ltn_part / ltn_w_part: le assiomi part-whole non esistono più nella versione 5")
    if cfg.ltn_norm not in ("mean", "region") or cfg.ltn_p < 0:
        raise ValueError(f"ltn_norm {cfg.ltn_norm!r} (mean|region), ltn_p {cfg.ltn_p} (0 = schedule, > 0 = fisso)")
    if cfg.ltn_w_cat != 1.0 or cfg.ltn_ax_domain != "all":
        raise ValueError("ltn_w_cat / ltn_ax_domain sono della versione 3: usa ltn_ax_guard (nessuna categoria)")
    if cfg.ltn and cfg.ltn_attr_loss != "none" and not cfg.ltn_attrs:
        raise ValueError(f"ltn_attr_loss={cfg.ltn_attr_loss!r} richiede --ltn_attrs true (le etichette dei concetti)")
    if cfg.ltn:
        LTNL.resolve_axioms(cfg.ltn_axioms)   # unknown names, Normal axioms, duplicates (also checked by install)
    if cfg.clahe and (cfg.clahe_clip <= 0 or cfg.clahe_grid < 1):
        raise ValueError(f"CLAHE: clahe_clip {cfg.clahe_clip} deve essere > 0, clahe_grid {cfg.clahe_grid} >= 1")

    def lr_at(epoch: int) -> float:
        """lr of the epoch: as StepLR called at the end of the epoch; depends only on the epoch, so it also holds on resume"""
        if cfg.lr_schedule == "step":
            return cfg.lr * cfg.lr_gamma ** (epoch // cfg.lr_step_size)
        return cfg.lr

    n_steps = len(train_loader)
    sched_txt = ("costante" if cfg.lr_schedule == "constant"
                 else f"x{cfg.lr_gamma} ogni {cfg.lr_step_size} epoche")
    print(f"{n_steps} step per epoca x {cfg.epochs} epoche = {n_steps * cfg.epochs} step "
          f"| {cfg.optimizer} lr {cfg.lr} {sched_txt} | batch {cfg.batch_size}", flush=True)

    best = dict(map_50=-1.0, epoch=-1)
    first_ids_by_epoch = []
    global_step = 0
    start_epoch = 0
    if cfg.resume:
        ck = torch.load(LAST_PT, map_location=DEVICE, weights_only=False)
        if ck.get("optimizer") is None:
            raise RuntimeError("last.pt non contiene lo stato dell'ottimizzatore (run creata dal notebook): "
                               "la ripresa non sarebbe equivalente a un training continuo")
        model.load_state_dict(ck["model"])
        optimiser.load_state_dict(ck["optimizer"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        done = []
        if os.path.exists(HISTORY_JSON):
            with open(HISTORY_JSON, "r", encoding="utf-8") as f:
                done = [h for h in json.load(f) if h["epoch"] < start_epoch]
        evaluated = [h for h in done if "val/map_50" in h]
        if evaluated:
            b = max(evaluated, key=lambda h: h["val/map_50"])
            best = dict(map_50=b["val/map_50"], epoch=b["epoch"])
        first_ids_by_epoch = [tuple(h["first_ids"]) for h in done]
        global_step = start_epoch * n_steps
        _set_rng_state(ck["rng"])
        print(f"RIPRESA da {LAST_PT}: epoca {ck['epoch']} completata, riparto dall'epoca {start_epoch} "
              f"| miglior mAP@50 finora {best['map_50']:.4f} (epoca {best['epoch']})", flush=True)
    try:
        for epoch in range(start_epoch, cfg.epochs):
            for g in optimiser.param_groups:
                g["lr"] = lr_at(epoch)
            if cfg.ltn:
                # LTN: ApME exponent for this epoch (v5 runs: fixed p = 2)
                model.roi_heads.ltn_p = (float(cfg.ltn_p) if cfg.ltn_p > 0                # ApME exponent: fixed,
                                         else LTNL.p_schedule(epoch, cfg.epochs))   # or 2 -> 10
            model.train()
            t0 = time.time()
            sums, nb, first_ids, ltn_n = defaultdict(float), 0, [], defaultdict(int)
            for step, (imgs, targets, ids) in enumerate( tqdm(train_loader, desc=f"epoca {epoch:>3} train", leave=False), start=1):

                if len(first_ids) < 5:
                    first_ids.extend(ids[:5 - len(first_ids)])
                imgs = [i.to(DEVICE, non_blocking=True) for i in imgs]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                with torch.autocast(device_type=DEVICE.type, enabled=(cfg.amp and DEVICE.type == "cuda")):
                    losses = model(imgs, targets)
                    total = sum(losses.values())
                if not torch.isfinite(total):
                    raise FloatingPointError(f"epoca {epoch} step {step}: loss non finita "
                                             f"{ {k: float(v) for k, v in losses.items()} }")
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                if cfg.grad_clip is not None:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                scaler.step(optimiser)
                scaler.update()

                for k, v in losses.items():
                    sums[f"train/{k}"] += float(v.detach())
                sums["train/loss"] += float(total.detach())
                if cfg.ltn:
                    # LTN: logged values of this batch, averaged later only over the batches where they exist
                    for k, v in model.roi_heads.last_log.items():   # satisfaction per axiom / head
                        sums[f"train/{k}"] += v
                        ltn_n[f"train/{k}"] += 1
                nb += 1
                global_step += 1
                if cfg.log_every and step % cfg.log_every == 0:
                    rate = step / max(time.time() - t0, 1e-9)
                    print(f"    epoca {epoch:>3} | step {step:>5}/{n_steps} | loss {sums['train/loss'] / step:.4f} "
                          f"| {rate:.1f} it/s | eta {(n_steps - step) / max(rate, 1e-9) / 60:.1f} min", flush=True)
                    wb.log({f"step/{k}": float(v.detach()) for k, v in losses.items()}
                           | {"step/loss": float(total.detach())}, step=global_step)

            row = {k: v / max(1, ltn_n[k] if k in ltn_n else nb) for k, v in sums.items()}   # see validation_losses
            row.update(epoch=epoch, lr=optimiser.param_groups[0]["lr"],
                       seconds=round(time.time() - t0, 1), first_ids=first_ids,
                       aug_applied=AUG_STATS["applied"], aug_refused=AUG_STATS["refused"],
                       aug_photo=AUG_STATS["photo"], aug_vflip=AUG_STATS["vflip"],
                       aug_noise=AUG_STATS["noise"], aug_erase=AUG_STATS["erase"],
                       aug_erase_refused=AUG_STATS["erase_refused"], aug_elastic=AUG_STATS["elastic"],
                       aug_elastic_refused=AUG_STATS["elastic_refused"])
            AUG_STATS.clear()   # per-epoch counters (7b)

            # check for the defect of 05-07: the training order must change between epochs
            first_ids_by_epoch.append(tuple(first_ids))
            if len(first_ids_by_epoch) >= 2 and first_ids_by_epoch[-1] == first_ids_by_epoch[-2]:
                raise RuntimeError(f"epoca {epoch}: stessi primi image_id dell'epoca precedente "
                                   f"{first_ids}. Il mescolamento non sta cambiando.")

            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                val = evaluate(model, val_loader)
                vloss = validation_losses(model, val_loss_loader, seed=cfg.seed)
                row.update({f"val/{k}": v for k, v in val.items()})
                row.update({f"val/{k}": v for k, v in vloss.items()})
                is_best = val["map_50"] > best["map_50"]
                if is_best:
                    best = dict(map_50=val["map_50"], epoch=epoch)
                    save_checkpoint(BEST_PT, model, epoch, val)
                print(f"epoca {epoch:>3} | train {row['train/loss']:.4f} | val {row['val/loss']:.4f} | "
                      f"mAP@50 {val['map_50']:.4f} | FROC massa {val['froc_mass_mean']:.4f} "
                      f"({val['froc_mass_mean_dedup']:.4f} dedup) | calc {val['froc_calc_mean']:.4f} "
                      f"({val['froc_calc_mean_dedup']:.4f} dedup) | AUC {val['image_auc_malignancy']:.4f} "
                      f"| {row['seconds']:.0f}s{'  <-- best' if is_best else ''}", flush=True)
                if cfg.ltn:
                    keys = sorted(k for k in row if k.startswith("val/") and
                                  (k.startswith("val/loss_attr") or k.startswith("val/aux_")
                                   or k in ("val/loss_class_sat", "val/ce_class", "val/loss_clin", "val/loss_find",
                                            "val/loss_ent", "val/loss_kl", "val/loss_rec")
                                   or k.startswith("val/sat_") or k.startswith("val/ce_attr")))
                    if keys:
                        print("          LTN val | " + " | ".join(f"{k[4:]} {row[k]:.3f}" for k in keys), flush=True)
                wb.log({k: v for k, v in row.items() if isinstance(v, (int, float))}, step=global_step)
            append_history(row)
            save_checkpoint(LAST_PT, model, epoch, {k: v for k, v in row.items() if k != "first_ids"},
                            optimiser=optimiser, scaler=scaler)
            write_run_manifest("in training", model=model, optimiser=optimiser, scaler=scaler, wb=wb)

            # early stopping: best["epoch"] is the last epoch that improved val mAP@50 (also after --resume)
            if cfg.patience > 0 and best["epoch"] >= 0 and epoch - best["epoch"] >= cfg.patience:
                print(f"STOP ANTICIPATO all'epoca {epoch}: mAP@50 non migliora da {cfg.patience} epoche "
                      f"(miglior {best['map_50']:.4f} all'epoca {best['epoch']})", flush=True)
                break
    finally:
        wb.finish()
    print(f"\nmiglior mAP@50 {best['map_50']:.4f} all'epoca {best['epoch']} -> {BEST_PT}")
    return best


RUN = True   # False to load a finished run without retraining
try:
    best = train(model) if RUN else None
except KeyboardInterrupt:
    write_run_manifest("interrotta (riprendibile con --resume true)")
    raise
except BaseException as e:
    write_run_manifest(f"errore: {type(e).__name__}: {e}")
    raise

# ====================================================================================================
# 14. Report from the best checkpoint (never from the last epoch)
# ====================================================================================================
ck = torch.load(BEST_PT, map_location=DEVICE, weights_only=False)
if ck["class_names"] != CLASS_NAMES:
    raise ValueError(f"vocabolario del checkpoint {ck['class_names']} != {CLASS_NAMES}")
model.load_state_dict(ck["model"])
PROVENANCE = f"{BEST_PT} | epoca {ck['epoch']} | salvato {ck['saved_at']}"

report = evaluate(model, make_loader(val_records, train=False))
report.update(validation_losses(model, make_loader(val_records, train=False,
                                                   batch_size=cfg.batch_size), seed=cfg.seed))

print("=" * 72)
print("REPORT DI VALIDATION  (mai test)")
print(PROVENANCE)
print(f"immagini {len(val_records)} | lesioni {sum(len(r['lesions']) for r in val_records)}")
print("-" * 72)
for k in ("map_50", "map_75", "map", "ap_mass_benign", "ap_mass_malignant",
          "ap_calc_benign", "ap_calc_malignant"):
    if k in report:
        print(f"  {k:<26} {report[k]:.4f}")
print()
print(f"  {'':<26} {'originale':>10} {'dedup':>10}")
for fam in ("mass", "calc"):
    print(f"  froc_{fam}_mean{'':<16} {report[f'froc_{fam}_mean']:>10.4f} "
          f"{report[f'froc_{fam}_mean_dedup']:>10.4f}")
print()
print(f"  {'image_auc_malignancy':<26} {report['image_auc_malignancy']:.4f}")
print(f"  {'n_detections':<26} {report['n_detections']}")
print(f"  {'val loss':<26} {report['loss']:.4f}")
print("=" * 72)

with open(os.path.join(RUN_DIR, "report_validation.json"), "w", encoding="utf-8") as f:
    json.dump(dict(provenance=PROVENANCE, config=asdict(cfg), report=report), f, indent=2)
print("scritto ->", os.path.join(RUN_DIR, "report_validation.json"))
write_run_manifest("completata")
print("manifest della run ->", RUN_MANIFEST_JSON)

# ====================================================================================================
# 15. Test
# ====================================================================================================
TEST_SET_LOCKED = True
print("TEST_SET_LOCKED =", TEST_SET_LOCKED, "| nessun numero di test in questo notebook")
