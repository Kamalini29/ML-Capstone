# Data

This folder holds the three raw datasets used by the notebooks. They are **not** committed
here yet — download each one and save it under the exact filename the notebooks expect:

| Track | Dataset | Source | Save as |
|---|---|---|---|
| Regression | Oil Well | https://www.kaggle.com/datasets/ruslanzalevskikh/oil-well | `data/oil_well.csv` |
| Classification | Google Play Store Apps | https://www.kaggle.com/datasets/yusufdelikkaya/google-play-store-apps-dataset | `data/playstore_apps.csv` |
| Clustering (Review 2) | Gene Expression Cancer RNA-Seq | https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq | `data/gene_expression_cancer.csv` (+ its labels file — UCI ships data and labels separately for this one) |

## How to download

**Kaggle datasets (oil well, play store):**
1. `pip install kaggle --break-system-packages`
2. Get an API token from Kaggle → Account → *Create New API Token* → save `kaggle.json` to `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   kaggle datasets download -d ruslanzalevskikh/oil-well -p data/ --unzip
   kaggle datasets download -d yusufdelikkaya/google-play-store-apps-dataset -p data/ --unzip
   ```
4. Rename whatever CSV comes out of the zip to `oil_well.csv` / `playstore_apps.csv` (Kaggle
   zips often contain a differently-named CSV inside).

**UCI dataset (gene expression, needed for Review 2 only):**
Download from the link above (it's a zip containing `data.csv` and `labels.csv`) and place
both files in this folder.

## Once downloaded

Open `notebooks/regression.ipynb`, run the first cell, and check the printed column list —
then set `TARGET_COLUMN` in the next cell to the real target column name before running the
rest of the notebook. Same pattern in `classification.ipynb` for `RATING_COL`.
