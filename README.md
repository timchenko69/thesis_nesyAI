# LTN on Faster R-CNN for CBIS-DDSM: version 5

This is the MSc thesis code of Temurbek Karimov (Politecnico di Torino). It adds a Logic Tensor Network to a Faster R-CNN
detector. The knowledge base is the one of SymMedNet (BMVC submission, appendix B.9), adapted to a detector.

## Repository layout

```
ltn_v5/
├── README.md               this file
├── requirements.txt        package versions of the lab environment
├── code/                   all Python code (flat imports: run every script from this folder)
│   ├── ltn_logic.py              the logic: connectives, ApME, groundings, facts φ, axioms ψ, losses 1 − Sat
│   ├── ltn_heads.py              LTNRoIHeads: torchvision's RoIHeads + concept heads + knowledge-base losses
│   ├── ltn_attributes.py         concept heads and vocabularies, CBIS encoding, reader of the facts file
│   ├── build_facts.py            writes facts_v5.csv (one row per train/val lesion, one column per fact φ1–φ21)
│   ├── train_frcnn.py            training script (the group's script; with --ltn false it is the baseline)
│   ├── ltn_examples_and_concepts.py   evaluation only: concept heads and axioms on the predictions, example pages
│   ├── test_ltn.py               47 unit tests: logic, facts file (CPU, no data)
│   └── test_ltn_heads.py         12 unit tests: the heads on a real Faster R-CNN (CPU, random weights)
├── notebooks/
│   └── 24_ltn_v5.ipynb     the whole pipeline WITH the outputs of the actual runs (sections 0–10)
├── knowledge_base/
│   └── LTN_axioms_v5.xlsx  heads, groundings, axioms in the paper's notation, comparison with SymMedNet, sources
└── results/                validation split only, seed 1337, 50 epochs
    ├── detection/          v5_best.csv (one row per run), v5_history.csv (per epoch), v5_map50.png
    └── concepts/           concept_eval.csv (concept heads), axiom_eval.csv (axioms on predictions),
        └── examples/       example pages: AxC vs C on G, the same 16 validation images
```

The code stays in one flat folder because the modules import each other by name (`import ltn_logic`), exactly as on
the lab machine, so nothing had to be changed to reorganise the repository.

## Paper → code

### Individuals and groundings (`ltn_logic.py`, module docstring)

- Individuals x are the RoIs sampled by the box head (512 per image). Each RoI gets the label of the matched lesion (IoU ≥ 0.5), or background.
- The class head is one softmax over 5 outputs: background, mass benign, mass malignant, calc benign, calc malignant.

| Paper | Detector |
|---|---|
| `isOfClass(x, ℓ_Benign)` | p(mass benign) + p(calc benign) |
| `isOfClass(x, ℓ_Malignant)` | p(mass malignant) + p(calc malignant) |
| `has_mass(x, c_mass)`, `has_calc(x, c_calc)` | sums of the class softmax (`class_predicates`) |
| `has_shape`, `has_distribution`, `is_high_density` | softmax heads (shape 3 outputs, distribution 5, density 2) |
| `has_margin`, `has_morphology` | sigmoid heads, 5 outputs each (multi-label, as in section 3.3 of the paper) |
| constants `c_α` | `CONSTANTS`: constant → (head, value) |

### Knowledge base (`kb_losses`). Every loss is `1 − Sat`

| Formula | Code | Loss |
|---|---|---|
| φ_class = ∀ Diag(x, ℓ_k). isOfClass(x, ℓ_k) | `class_facts_sat` | `loss_class_sat` |
| φ1, φ2 = ∀ Diag(x, l_mass). has_mass(x, c_mass) / calc | `finding_facts_sat` | `loss_find` |
| φ3–φ21 = ∀ Diag(x, l_α). has_attr(x, l_α) | `head_facts_sat`, facts from `facts_v5.csv` | `loss_attr` |
| ψ3–ψ14, ψ16 = ∀ Diag(x, ℓ_k): body ⇒ isOfClass(x, ℓ_k) | `AXIOMS`, `axiom_truths`, `axiom_domain` | `loss_clin` |

- `axiom_text(name)` prints each axiom in the paper's notation. Every run also prints them at the start of its log.
- ψ1, ψ2 and ψ15 conclude Normal. The detector has no Normal class, so they are refused (call of 6 November).

### Adaptations to the detector

1. **Aggregator of φ_class.** Almost all RoIs are background. With one ApME clause the calcifications disappeared (FROC 0); with one ApME clause per class the background weighed 1/5 and the model produced 7× more detections. Two aggregators keep 1 − Sat:
   - `geo`: the geometric mean, i.e. the product over the regions, as in Faster-LTN. Here 1 − Sat = 1 − exp(−CE).
   - `balanced`: ApME weighted 1/2 background and 1/8 per lesion class, as SymMedNet's batch balance.

   Block 1 of the notebook chooses between them.
2. **Weight of each term.** Each fact or axiom term is weighted by the share of RoIs in its domain (`norm="region"`), like torchvision's box loss.
3. **p of ApME.** Fixed at p = 2 for the 50-epoch runs. SymMedNet uses the schedule 2 → 10 over 150 epochs (`p_schedule`).
4. **Negative facts.** CBIS lists every value of a lesion, so for the sigmoid heads (margin, morphology) the values that are not listed become negative facts.
5. **has_mass and the family.** By default, as in the paper, has_mass(x, c_mass) in an axiom is predicted by the model (`ax_family="pred"`). `ax_family="label"` instead fixes the family from the RoI's label: the class head gives family and class from one softmax, and with a predicted family an axiom can be satisfied by moving "malignant mass" to "malignant calcification" (unit test `test_predicted_family_as_in_the_paper_can_move_mass_to_calc`).

## How to run

1. Open `notebooks/24_ltn_v5.ipynb`, set `CODE_DIR` (cell 0) to the `code/` folder and the data paths, run the cells
   in order. The notebook already contains the outputs of the runs reported in `results/`.
2. Or by hand, from `code/`:
   - `python build_facts.py` writes `F:\runs\facts_v5\facts_v5.csv` (train/val lesions only; the test split is never read).
   - `python train_frcnn.py --config F:\runs\E45_e100_step60\run_manifest.json --run_name v5_G_AxC_geo_e50 --epochs 50 --lr_step_size 60 --ltn true --ltn_class_loss sat --ltn_class_agg geo --ltn_attrs true --ltn_facts_file F:\runs\facts_v5\facts_v5.csv --ltn_attr_loss facts --ltn_axioms kb16 --ltn_norm region --ltn_p 2`
3. Tests, from `code/`: `python test_ltn.py` (47 tests) and `python test_ltn_heads.py` (12 tests).
4. Concept evaluation of the trained runs: `python ltn_examples_and_concepts.py` (paths in its CONFIG block), or
   section 10 of the notebook.

Not in the repository (lab machine, `F:\runs`): the checkpoints (`best.pt`, `last.pt`), `history.json` and
`run_manifest.json` of every run, the console logs, the facts file `facts_v5.csv`, the CBIS-DDSM images.

## Environment

Windows 10, one NVIDIA RTX 3090 (shared), Python 3.10, PyTorch 2.1.2 + CUDA 12.1, torchvision 0.16.2 (see
`requirements.txt`). Seed 1337; the test split is never read by any of these files.

## Where to look first in the code (for a review)

| What | Where |
|---|---|
| Groundings | `ltn_logic.class_predicates`, `literal`, `CONSTANTS`, `PREDICATE_HEAD` |
| Class facts and aggregators | `ltn_logic.class_facts_sat` |
| Axioms as in the paper | `ltn_logic.AXIOMS`, `axiom_text`, `axiom_truths`, `axiom_domain` |
| All losses of one batch | `ltn_logic.kb_losses` |
| Where the losses enter the detector | `ltn_heads.LTNRoIHeads.forward`, `_ltn_losses` |
| Concept targets of the RoIs | `ltn_heads.gather_attr_targets` |
| Facts file | `build_facts.py`, `ltn_attributes.facts_row`, `load_facts_file` |
| Concept evaluation (matching, metrics, axioms on predictions) | `ltn_examples_and_concepts.evaluate`, `concept_metrics`, `axiom_metrics` |
