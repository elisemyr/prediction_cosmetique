"""
Extrait deux types de features à partir des images produit :
1. Features manuelles : couleurs dominantes, contraste, ratio de texte (OCR)
2. Embeddings CV : CLIP pré-entraîné avec transformers

Commande : uv run scripts/extract_features.py
"""

import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = "data/processed/dataset_clean.parquet"
OUTPUT_MANUAL = "data/processed/features_manual.parquet"
OUTPUT_EMBEDDINGS = "data/processed/embeddings_clip.npy"
OUTPUT_IDS = "data/processed/embeddings_ids.npy"


# ---------- BRANCHE 1 : FEATURES MANUELLES ----------

def get_dominant_colors(img, n_colors=3):
    """Retourne les n couleurs dominantes (RGB) via k-means sur les pixels."""
    img_small = img.resize((100, 100))
    pixels = np.array(img_small).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    # trie par taille de cluster (couleur la plus fréquente en premier)
    counts = np.bincount(kmeans.labels_)
    order = np.argsort(-counts)
    return colors[order]


def get_contrast(img):
    """Écart-type des niveaux de gris = proxy simple de contraste global"""
    gray = np.array(img.convert("L"))
    return float(gray.std())


def get_brightness(img):
    """Luminosité moyenne"""
    gray = np.array(img.convert("L"))
    return float(gray.mean())


def get_saturation(img):
    """Saturation moyenne (HSV)"""
    hsv = np.array(img.convert("HSV"))
    return float(hsv[:, :, 1].mean())


def extract_manual_features(row):
    try:
        img = Image.open(row["image_path"]).convert("RGB")
        colors = get_dominant_colors(img, n_colors=3)

        return {
            "id": row["id"],
            "dominant_r": colors[0][0],
            "dominant_g": colors[0][1],
            "dominant_b": colors[0][2],
            "second_r": colors[1][0],
            "second_g": colors[1][1],
            "second_b": colors[1][2],
            "contrast": get_contrast(img),
            "brightness": get_brightness(img),
            "saturation": get_saturation(img),
            "aspect_ratio": img.width / img.height,
        }
    except Exception as e:
        print(f"Erreur features manuelles id={row['id']}: {e}")
        return None


def run_manual_features(df):
    print("Extraction des features manuelles (couleurs, contraste, luminosité)...")
    results = []
    for i, row in df.iterrows():
        feats = extract_manual_features(row)
        if feats is not None:
            results.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)} traités")

    df_feats = pd.DataFrame(results)
    df_feats.to_parquet(OUTPUT_MANUAL, index=False)
    print(f"Features manuelles sauvegardées : {OUTPUT_MANUAL} ({len(df_feats)} lignes)")
    return df_feats


# ---------- BRANCHE 2 : EMBEDDINGS CLIP ----------

def _image_features(model, inputs):
    """transformers >= 5 renvoie un BaseModelOutputWithPooling, les versions
    antérieures un tenseur directement."""
    out = model.get_image_features(**inputs)
    return out if hasattr(out, "cpu") else out.pooler_output


def run_clip_embeddings(df, batch_size=32):
    print("\nChargement du modèle CLIP (peut prendre un moment au premier lancement)...")
    import torch
    from transformers import CLIPProcessor, CLIPModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"  # mps = GPU Apple Silicon
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    embeddings = []
    ids = []
    n_errors = 0

    print(f"Extraction des embeddings CLIP sur {device} (batch={batch_size})...")
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]

            images = []
            batch_ids = []
            for _, row in batch.iterrows():
                try:
                    images.append(Image.open(row["image_path"]).convert("RGB"))
                    batch_ids.append(row["id"])
                except Exception as e:
                    n_errors += 1
                    print(f"Erreur lecture image id={row['id']}: {e}")

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt").to(device)
            embeddings.append(_image_features(model, inputs).cpu().numpy())
            ids.extend(batch_ids)

            print(f"  {min(start + batch_size, len(df))}/{len(df)} traités")

    if not embeddings:
        raise RuntimeError("Aucun embedding extrait, rien à sauvegarder.")

    embeddings = np.vstack(embeddings)
    ids = np.array(ids)

    np.save(OUTPUT_EMBEDDINGS, embeddings)
    np.save(OUTPUT_IDS, ids)
    print(f"Embeddings sauvegardés : {OUTPUT_EMBEDDINGS} (shape {embeddings.shape}, {n_errors} images ignorées)")


if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    print(f"{len(df)} produits chargés.\n")

    run_manual_features(df)
    run_clip_embeddings(df)

    print("\nTerminé, Features manuelles + embeddings CLIP disponibles dans data/processed/")