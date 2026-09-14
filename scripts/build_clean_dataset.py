"""
Nettoie et joint le CSV beauty_data.csv avec les chemins d'images locales
Produit une table finale prête pour l'EDA et la modélisation

Commande: uv run scripts/build_clean_dataset.py
"""

import pandas as pd
import ast
import os

RAW_CSV = "data/raw/beauty_data.csv"
IMAGES_DIR = "data/images"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_clean.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_raw():
    print("Chargement de beauty_data.csv...")
    df = pd.read_csv(RAW_CSV)
    print(f"{len(df)} lignes chargées, colonnes : {df.columns.tolist()}")
    return df


def check_image_exists(row):
    """Vérifie que le fichier image existe réellement sur disque."""
    if pd.isna(row["image_name"]):
        return False
    path = os.path.join(IMAGES_DIR, row["image_name"])
    return os.path.exists(path)


def build_clean_dataset(df):
    print("Vérification de l'existence des images...")
    df["image_exists"] = df.apply(check_image_exists, axis=1)

    n_missing = (~df["image_exists"]).sum()
    print(f"{n_missing} lignes sans image exploitable sur {len(df)} — elles seront exclues.")

    df_clean = df[df["image_exists"]].copy()

    # Chemin complet vers l'image, utile pour la suite (feature extraction)
    df_clean["image_path"] = df_clean["image_name"].apply(lambda x: os.path.join(IMAGES_DIR, x))

    # Colonnes cibles à garder (les contains_* sont déjà des booléens propres)
    target_cols = [
        "contains_fragrance",
        "contains_drying_alcohol",
        "contains_parabens",
        "contains_sulfates",
        "contains_silicones",
    ]

    # Colonnes descriptives utiles pour l'EDA / stratification
    keep_cols = ["id", "brand", "name", "image_path", "category", "origin"] + target_cols

    df_final = df_clean[keep_cols].copy()

    # category et origin peuvent être null — on garde explicite plutôt que de deviner
    df_final["category"] = df_final["category"].fillna("unknown")
    df_final["origin"] = df_final["origin"].fillna("unknown")

    return df_final, target_cols


def print_summary(df_final, target_cols):
    print("\n--- Résumé du dataset final ---")
    print(f"Nombre de produits : {len(df_final)}")
    print(f"\nDistribution des catégories :\n{df_final['category'].value_counts()}")

    print("\nDistribution des targets (proportion de True) :")
    for col in target_cols:
        pct = df_final[col].mean() * 100
        print(f"  {col}: {pct:.1f}%")


if __name__ == "__main__":
    df_raw = load_raw()
    df_final, target_cols = build_clean_dataset(df_raw)
    print_summary(df_final, target_cols)

    df_final.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSauvegardé dans {OUTPUT_PATH}")