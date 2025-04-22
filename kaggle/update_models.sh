#!/usr/bin/env bash
# ─── CONFIG ────────────────────────────────────────────────────────────────
# Weights & Biases config
ENTITY="BashHav"                      
PROJECT="audio-classification"        

RUN_ID="tjpu9gbf"  
VERSION="v9"                                  # replace with the run ID
ARTIFACT="model-${RUN_ID}:${VERSION}"         # replace artifact name:version
                   

BASE_DIR="data/kaggle_datasets"
DATASET_NAME="birdclef2025_models"
MAIN_FOLDER="main_folder"

# Kaggle config for saving old versions to locan main folder
KAGGLE_DATASET_NAME="ivanbashtovyi/birdclef2025-models"
KAGGLE_DEST_DIR="${BASE_DIR}/${DATASET_NAME}/${MAIN_FOLDER}"

# ─── 2) PREP ───────────────────────────────────────────────────────────────
echo ">>> Preparing directory structure"
rm -rf "$KAGGLE_DEST_DIR"
mkdir -p "$KAGGLE_DEST_DIR"

kaggle datasets download \
  -d "$KAGGLE_DATASET_NAME" \
  -p "$KAGGLE_DEST_DIR" \
  --unzip \
  --force

echo "✅ Updated $KAGGLE_DEST_DIR"


# ─── 1) DERIVE DESTINATION FOLDER FROM RUN NAME ────────────────────────────
RUN_NAME=$(python3 <<EOF
import wandb, re
ENTITY="${ENTITY}"
PROJECT="${PROJECT}"
RUN_ID="${RUN_ID}"
api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
name = run.name or run.id
# slugify: replace non‑alphanumeric with underscore
slug = re.sub(r"\W+", "_", name)
print(slug)
EOF
)

DEST_DIR="${BASE_DIR}/${DATASET_NAME}/${MAIN_FOLDER}/${RUN_ID}_${RUN_NAME}"

# # ─── 2) PREP ───────────────────────────────────────────────────────────────
# echo ">>> Preparing directory structure"
# rm -rf "$DEST_DIR"
# mkdir -p "$DEST_DIR"

# ─── 3) DOWNLOAD ARTIFACT ─────────────────────────────────────────────────
echo ">>> Downloading artifact into $DEST_DIR"
wandb artifact get "$ENTITY/$PROJECT/$ARTIFACT" --root "$DEST_DIR"

# ─── 4) DUMP RUN CONFIG ───────────────────────────────────────────────────
echo ">>> Saving run config to $DEST_DIR/config.yaml"
python3 <<EOF
import os, wandb, yaml
ENTITY="${ENTITY}"
PROJECT="${PROJECT}"
RUN_ID="${RUN_ID}"
DEST="${DEST_DIR}"
api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
cfg = dict(run.config)
with open(os.path.join(DEST, "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
EOF

# ─── 5) UPLOAD TO KAGGLE ─────────────────────────────────────────────────
cd "${BASE_DIR}"

echo ">>> Packaging '${DATASET_NAME}/${MAIN_FOLDER}' for Kaggle upload"
kaggle datasets version \
  -p "${DATASET_NAME}" \
  -m "Add model+config for ${RUN_NAME}" \
  --dir-mode zip


echo "Cleaning main folder"

DATASET_NAME="birdclef2025_models"
MAIN_FOLDER="main_folder"

CLEAN_PATH="${DATASET_NAME}/${MAIN_FOLDER}"
rm -rf "${CLEAN_PATH:?}/"*

echo "✅ Done! Model + config have been downloaded and uploaded to Kaggle."




