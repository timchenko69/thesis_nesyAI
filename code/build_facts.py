"""
Builds the facts file of the LTN (version 5): facts_v5.csv, one row per lesion of the TRAIN and VALIDATION split,
one column per fact phi1-phi21 of SymMedNet (appendix B.9.1). train_frcnn.py reads the facts only from this file.

Inputs (the same files as train_frcnn.py):
  * the group's image-level file (cbis_ddsm_image_level_combined.xlsx): split, image label (label_upd), mass density
  * the manifest of the PNG masks (manifest.csv): the lesions of every image (image_id, lesion_id), status ok
  * the official CBIS-DDSM case-description CSVs (TRAINING files): shape, margin, calcification type and distribution,
    joined by (image folder, abnormality id) = (image_id, lesion_id)
The test split is never read: images are taken from the split file only if their split is Train or Valid, and the CBIS
rows of any other image are dropped on reading (ltn_attributes.load_cbis_csvs).

Cell values: 1 = fact true, 0 = false (for a sigmoid head the value is not listed; for a softmax head another value is),
empty = unknown (no fact for that head). Compound CBIS values: softmax heads keep the most suspicious value (ties stay
1 = disjunction), sigmoid heads (margin, morphology) mark every listed value. Density only on single-lesion images.

Usage (lab machine, the mammoltn environment; a few seconds):
    python build_facts.py
    python build_facts.py --out F:\\runs\\facts_v5\\facts_v5.csv
Writes the CSV, a summary (facts_v5_summary.csv: facts per column and split) and prints a check of the CSV join
(pathology of the CBIS row = image label).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ltn_attributes as A          # noqa: E402

DEFAULTS = dict(data_root=r"F:\MammoLTN-full-images", png_dir_name="CBIS-DDSM-PNG16-CROPPED-MASKS",
                splits_dir_name="CBIS_08_11_2024", combined_xlsx_name="cbis_ddsm_image_level_combined.xlsx",
                csv_files=("mass_case_description_train_set.csv", "calc_case_description_train_set.csv"),
                out=r"F:\runs\facts_v5\facts_v5.csv")
SPLIT_MAP = {"Train": "train", "Valid": "val"}          # "Test" is never mapped: those rows are dropped at once


def image_id_from_filename(fn: str) -> str:
    fn = str(fn)
    return fn[:-9] if fn.endswith("_crop.png") else os.path.splitext(fn)[0]


def build(data_root, png_dir_name, splits_dir_name, combined_xlsx_name, csv_dir, csv_files):
    xlsx = os.path.join(data_root, splits_dir_name, combined_xlsx_name)
    manifest_csv = os.path.join(data_root, png_dir_name, "manifest.csv")
    csv_paths = [os.path.join(csv_dir, f) for f in csv_files]
    for p in [xlsx, manifest_csv, *csv_paths]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    df = pd.read_excel(xlsx)
    df.columns = [c.strip() for c in df.columns]
    A.check_columns(df.columns)
    df = df[df["split"].isin(SPLIT_MAP)].copy()                          # test rows dropped here, never used
    df["image_id"] = df["image_filename"].map(image_id_from_filename)
    mixed = df["source_patch_labels"].astype(str).map(lambda s: "BENIGN" in s and "MALIGNANT" in s)
    df = df[~mixed]                                                      # as train_frcnn.py: mixed pathology out
    df["split_norm"] = df["split"].map(SPLIT_MAP)

    index, kept = A.load_cbis_csvs(csv_paths, set(df["image_id"]))
    raw_path = {}
    for p in csv_paths:                                                  # pathology of the CBIS row, for the check
        c = pd.read_csv(p)
        c.columns = [x.strip() for x in c.columns]
        c["image_id"] = c["image file path"].astype(str).str.split("/").str[0].str.strip()
        c = c[c["image_id"].isin(set(df["image_id"]))]
        for _, r in c.iterrows():
            raw_path[(r["image_id"], int(r["abnormality id"]))] = str(r["pathology"]).strip().upper()

    man = pd.read_csv(manifest_csv)
    man.columns = [c.strip() for c in man.columns]
    man = man[man["status"].astype(str).str.lower() == "ok"]
    by_img = {k: v for k, v in man.groupby("image_id")}

    rows = []
    for _, r in df.iterrows():
        iid = r["image_id"]
        mrows = by_img.get(iid)
        if mrows is None:
            continue
        # family from the image name, as in train_frcnn.py
        fam = "MASS" if iid.startswith("Mass") else "CALC"
        pathology = str(r["label_upd"]).strip().upper()
        lids = [int(x) for x in mrows["lesion_id"].dropna()]
        # one row per lesion of the manifest, also without a CBIS row (then its facts are empty)
        for lid in lids:
            rec = index.get((iid, lid))
            row = A.facts_row(iid, lid, r["split_norm"], fam, pathology, len(lids), rec, r)
            row["cbis_pathology"] = raw_path.get((iid, lid), "")
            rows.append(row)
    out = pd.DataFrame(rows)
    cols = A.ID_COLUMNS + ["cbis_pathology"] + list(A.FINDING_COLUMNS) + [c for c, _, _ in A.FACT_COLUMNS]
    return out[cols].sort_values(["split", "image_id", "lesion_id"]).reset_index(drop=True), kept


def summary(facts: pd.DataFrame) -> pd.DataFrame:
    """Per split and fact column: lesions with the fact known, true, false."""
    out = []
    for sp, d in facts.groupby("split"):
        for c in list(A.FINDING_COLUMNS) + [c for c, _, _ in A.FACT_COLUMNS]:
            v = d[c].astype(str)
            out.append(dict(split=sp, fact=c, known=int((v != "").sum()), true=int((v == "1").sum()),
                            false=int((v == "0").sum())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="facts_v5.csv for the LTN (train / val lesions only)")
    for k in ("data_root", "png_dir_name", "splits_dir_name", "combined_xlsx_name", "out"):
        ap.add_argument(f"--{k}", default=DEFAULTS[k])
    ap.add_argument("--csv_dir", default="", help='folder of the CBIS CSVs; "" = data_root/splits_dir_name')
    a = ap.parse_args()
    csv_dir = a.csv_dir or os.path.join(a.data_root, a.splits_dir_name)
    facts, kept = build(a.data_root, a.png_dir_name, a.splits_dir_name, a.combined_xlsx_name, csv_dir,
                        DEFAULTS["csv_files"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    facts.to_csv(a.out, index=False)
    summ = summary(facts)
    summ.to_csv(os.path.splitext(a.out)[0] + "_summary.csv", index=False)

    print(f"CBIS CSV rows kept (train/val images): {kept}")
    for sp, d in facts.groupby("split"):
        print(f"{sp}: {len(d)} lesions on {d['image_id'].nunique()} images | {int((~d['in_cbis_csv']).sum())} without a "
              f"CBIS row | masses {int((d['family'] == 'MASS').sum())}, calcifications {int((d['family'] == 'CALC').sum())}")
    j = facts[facts["cbis_pathology"] != ""]
    # BENIGN_WITHOUT_CALLBACK counts as benign, as in the image labels
    agree = (j["cbis_pathology"].str.replace("BENIGN_WITHOUT_CALLBACK", "BENIGN") == j["pathology"]).mean()
    print(f"check of the join: pathology of the CBIS row = image label for {agree:.1%} of {len(j)} lesions")
    print(f"\nwritten {a.out} ({len(facts)} rows) and its _summary.csv")
    if agree < 0.97:
        raise SystemExit("the CBIS rows do not match the lesions: check the join before training")


if __name__ == "__main__":
    main()
