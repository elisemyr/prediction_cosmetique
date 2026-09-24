"""
Prépare un fichier JSON compact pour l'interface web :
- résultats AUC des 5 attributs
- SHAP + coefficients catégorie pour contains_sulfates
- prédictions out-of-fold (cross-val) pour l'explorateur produit
- miniatures d'images encodées en base64

Run le script: uv run scripts/export_for_interface.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from PIL import Image
import shap
import base64
import io
import json
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGETS = ["contains_fragrance", "contains_drying_alcohol", "contains_parabens",
           "contains_sulfates", "contains_silicones"]
THUMB_SIZE = (110, 110)
JPEG_QUALITY = 55

df_meta = pd.read_parquet("data/processed/dataset_clean.parquet")
df_manual = pd.read_parquet("data/processed/features_manual.parquet")
embeddings = np.load("data/processed/embeddings_clip.npy")
embedding_ids = np.load("data/processed/embeddings_ids.npy")

df_emb = pd.DataFrame(embeddings, columns=[f"clip_{i}" for i in range(embeddings.shape[1])])
df_emb["id"] = embedding_ids
df_full = df_meta.merge(df_manual, on="id").merge(df_emb, on="id")

category_dummies = pd.get_dummies(df_full["category"], prefix="cat")
df_full = pd.concat([df_full, category_dummies], axis=1)
cat_cols = category_dummies.columns.tolist()
manual_cols = ["dominant_r", "dominant_g", "dominant_b", "second_r", "second_g", "second_b",
               "contrast", "brightness", "saturation", "aspect_ratio"]
clip_cols = [c for c in df_full.columns if c.startswith("clip_")]


# ---------- 1. Résultats AUC des 5 attributs (train/test split, comme le notebook 02b) ----------

def run_pipeline_for_target(target, df_full):
    y = df_full[target].astype(int)
    idx_train, idx_test = train_test_split(df_full.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    out = {"pct_true": round(y.mean() * 100, 1)}

    m = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    m.fit(df_full.loc[idx_train, cat_cols], y_train)
    out["category_only"] = round(roc_auc_score(y_test, m.predict_proba(df_full.loc[idx_test, cat_cols])[:, 1]), 3)

    m = GradientBoostingClassifier(random_state=RANDOM_STATE)
    m.fit(df_full.loc[idx_train, manual_cols], y_train)
    out["manual"] = round(roc_auc_score(y_test, m.predict_proba(df_full.loc[idx_test, manual_cols])[:, 1]), 3)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(df_full.loc[idx_train, clip_cols])
    Xte = scaler.transform(df_full.loc[idx_test, clip_cols])
    m = LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE)
    m.fit(Xtr, y_train)
    out["clip"] = round(roc_auc_score(y_test, m.predict_proba(Xte)[:, 1]), 3)

    scaler2 = StandardScaler()
    Xtr2 = scaler2.fit_transform(df_full.loc[idx_train, clip_cols + cat_cols])
    Xte2 = scaler2.transform(df_full.loc[idx_test, clip_cols + cat_cols])
    m = LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE)
    m.fit(Xtr2, y_train)
    out["clip_plus_category"] = round(roc_auc_score(y_test, m.predict_proba(Xte2)[:, 1]), 3)

    return out

print("Calcul des résultats AUC pour les 5 attributs")
summary = {t.replace("contains_", ""): run_pipeline_for_target(t, df_full) for t in TARGETS}


# ---------- 2. SHAP + coefficients catégorie pour contains_sulfates ----------

print("Calcul SHAP et coefficients catégorie (contains_sulfates)...")
TARGET = "contains_sulfates"
y = df_full[TARGET].astype(int)
idx_train, idx_test = train_test_split(df_full.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

model_manual = GradientBoostingClassifier(random_state=RANDOM_STATE)
model_manual.fit(df_full.loc[idx_train, manual_cols], y_train)
explainer = shap.TreeExplainer(model_manual)
shap_values = explainer.shap_values(df_full.loc[idx_test, manual_cols])
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_order = np.argsort(-mean_abs_shap)

shap_export = {
    "features": [manual_cols[i] for i in shap_order],
    "mean_abs_shap": [round(float(mean_abs_shap[i]), 4) for i in shap_order],
}

scaler = StandardScaler()
Xtr = scaler.fit_transform(df_full.loc[idx_train, clip_cols + cat_cols])
Xte = scaler.transform(df_full.loc[idx_test, clip_cols + cat_cols])
model_ctrl = LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE)
model_ctrl.fit(Xtr, y_train)

coefs = pd.Series(model_ctrl.coef_[0], index=clip_cols + cat_cols)
cat_coefs = coefs[cat_cols].sort_values()
category_coefs_export = {
    "categories": [c.replace("cat_", "") for c in cat_coefs.index],
    "coef": [round(float(v), 3) for v in cat_coefs.values],
}


# ---------- 3. Prédictions out-of-fold (cross-val) pour l'explorateur ----------

print("Calcul des prédictions out-of-fold (5-fold cross-val) pour l'explorateur...")
X_ctrl_full = df_full[clip_cols + cat_cols]
scaler_full = StandardScaler()
X_ctrl_full_scaled = scaler_full.fit_transform(X_ctrl_full)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_proba = cross_val_predict(
    LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
    X_ctrl_full_scaled, y, cv=cv, method="predict_proba"
)[:, 1]

df_full["pred_proba_sulfates"] = oof_proba


# ---------- 4. Miniatures + export final ----------

print("Génération des miniatures (peut prendre 1-2 minutes)...")

def encode_thumbnail(path):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(THUMB_SIZE)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"Erreur miniature {path}: {e}")
        return None

products = []
for i, row in df_full.iterrows():
    thumb = encode_thumbnail(row["image_path"])
    if thumb is None:
        continue
    products.append({
        "id": int(row["id"]),
        "brand": row["brand"],
        "name": row["name"],
        "category": row["category"],
        "actual": {t.replace("contains_", ""): bool(row[t]) for t in TARGETS},
        "pred_proba_sulfates": round(float(row["pred_proba_sulfates"]), 3),
        "thumb": thumb,
    })
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(df_full)}")

output = {
    "summary": summary,
    "shap_sulfates": shap_export,
    "category_coefs_sulfates": category_coefs_export,
    "products": products,
}

with open("data/processed/interface_data.json", "w") as f:
    json.dump(output, f)

print(f"\nExport terminé : data/processed/interface_data.json ({len(products)} produits)")