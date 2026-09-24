# Le packaging trahit-il la formule ?

Projet portfolio explorant si l'image du packaging d'un produit cosmétique permet de prédire des attributs de sa composition — parfum, silicones, sulfates, parabènes, alcool asséchant — à partir de computer vision (embeddings CLIP) et de features visuelles classiques (couleur, contraste)

**[Voir l'interface interactive des résultats](https://elisemyr.github.io/prediction_cosmetique/)**

---

## Question de recherche

Le design visuel d'un packaging cosmétique reflète-t-il sa formulation ? Et si oui, ce signal visuel apporte-t-il une information au-delà de ce que révèle déjà la simple catégorie du produit (skincare, haircare, makeup...) ?

## Résultats en bref

| Attribut | % présent | Catégorie seule | Features manuelles | CLIP seul | CLIP + catégorie |
|---|---|---|---|---|---|
| Parfum | 58.0% | 0.558 | 0.486 | 0.542 | 0.550 |
| Alcool asséchant | 11.5% | 0.641 | 0.467 | 0.480 | 0.492 |
| Parabènes | 6.6% | 0.573 | 0.534 | 0.566 | 0.566 |
| **Sulfates** | 5.2% | 0.716 | 0.699 | 0.725 | **0.743** |
| Silicones | 24.9% | 0.644 | 0.468 | 0.578 | 0.611 |

*(AUC-ROC sur jeu de test, 20% des 1000 produits, stratifié)*

**Sulfates** est le seul attribut pour lequel le signal visuel (CLIP) apporte une information mesurable au-delà de la catégorie. Pour les quatre autres attributs, connaître la catégorie du produit égale ou dépasse ce que l'image apporte — un résultat négatif informatif plutôt qu'un échec : la composition d'un produit cosmétique ne se lit en général pas sur son emballage, sauf exception.

L'analyse SHAP sur l'attribut sulfates montre que le signal visuel capté (couleur dominante, saturation) recoupe en grande partie un indice de catégorie (haircare → sulfates ; skincare/suncare → sans sulfates) plutôt qu'une lecture fine et indépendante du packaging — un résultat nuancé documenté dans le notebook `04_explicabilite_shap.ipynb`.

## Choix méthodologiques et pivots

Ce projet a connu plusieurs pivots, documentés ici par souci de transparence méthodologique :

1. **Source de données** — l'objectif initial était de prédire le succès commercial d'un produit à partir de son packaging (catalogue Sephora). Après plusieurs tentatives (Kaggle sans images exploitables, reconstruction d'URL infructueuse, contenu chargé en JavaScript côté client empêchant un scraping minimal), le projet s'est réorienté vers **The Beauty API** (échantillon public Hugging Face, licence CC BY-NC 4.0), qui fournit des images normalisées et des données de composition (INCI) — mais aucune donnée de vente ou de notation
2. **Question de recherche** — ce changement de source a naturellement fait évoluer la question : de « prédire le succès commercial » à « prédire la formulation depuis le packaging », un sujet tout aussi pertinent et mieux servi par les données disponibles
3. **Attribution des résultats** — plutôt que de chercher un seul modèle "gagnant", le projet compare systématiquement 4 niveaux de modèles (catégorie seule, features manuelles, CLIP, CLIP+catégorie) pour chaque attribut, afin de distinguer un vrai signal visuel d'un simple proxy de catégorie

## Architecture du projet

```
prediction_cosmetique/
├── data/
│   ├── raw/                      # métadonnées brutes (beauty_data.csv)
│   ├── images/                   # 1000 images produit (.jpeg)
│   ├── processed/                # dataset nettoyé, features, embeddings
│   └── external/
├── scripts/
│   ├── download_hf_dataset.py    # téléchargement images + métadonnées (Hugging Face)
│   ├── build_clean_dataset.py    # nettoyage, jointure, construction des targets
│   ├── extract_features.py       # features manuelles (couleur/contraste) + embeddings CLIP
│   └── export_for_interface.py   # export des résultats pour l'interface interactive
├── notebooks/
│   ├── 01_EDA.ipynb                    # exploration : distributions, heatmap catégorie × attribut
│   ├── 02_baseline_features.ipynb      # validation du pipeline de modélisation (contains_fragrance)
│   ├── 02b_baseline_silicones.ipynb    # même pipeline (contains_silicones)
│   ├── 02c_all_targets.ipynb           # pipeline généralisé aux 5 attributs
│   └── 04_explicabilite_shap.ipynb     # SHAP + analyse des coefficients (contains_sulfates)
├── docs/
│   └── index.html                # interface interactive (GitHub Pages)
├── requirements.txt
└── README.md
```

## Données

- **Source** : [The Beauty API](https://huggingface.co/datasets/thebeautyapi/beautyproducts) — échantillon public de 1000 produits, licence **CC BY-NC 4.0** (usage non commercial)
- **Champs utilisés** : image du packaging, marque, nom, catégorie, et 5 indicateurs booléens de composition (`contains_fragrance`, `contains_drying_alcohol`, `contains_parabens`, `contains_sulfates`, `contains_silicones`).
- Ce dataset ne contient aucune donnée de vente, de notation ou de popularité — une limite assumée qui a motivé le pivot de la question de recherche

## Méthodologie

**Features visuelles**
- *Manuelles* : couleurs dominante et secondaire (RGB), contraste, luminosité, saturation, ratio largeur/hauteur — extraites via k-means et statistiques d'image simples.
- *Embeddings CLIP* : représentation à 512 dimensions par image, via `openai/clip-vit-base-patch32` (Hugging Face `transformers`)

**Modélisation**, pour chaque attribut, 4 modèles comparés sur un split train/test stratifié (80/20) :
1. Catégorie seule (régression logistique) — référence sans aucune image
2. Features manuelles (gradient boosting)
3. Embeddings CLIP seuls (régression logistique régularisée, `C=0.1`).
4. Embeddings CLIP + catégorie combinés

Le déséquilibre de classes (certains attributs à moins de 7% de positifs) impose l'usage de l'AUC-ROC plutôt que l'accuracy comme métrique principale.

**Explicabilité** : SHAP (`TreeExplainer`) sur le modèle à features manuelles, et analyse directe des coefficients de régression logistique pour le poids des catégories — les embeddings CLIP bruts n'étant pas interprétables terme à terme, aucune tentative de leur appliquer SHAP dimension par dimension n'a été faite

## Reproduire le projet

```bash
git clone https://github.com/elisemyr/prediction_cosmetique.git
cd prediction_cosmetique
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

uv run scripts/download_hf_dataset.py
uv run scripts/build_clean_dataset.py
uv run scripts/extract_features.py
```

Puis exécuter les notebooks dans l'ordre (`01` → `02`→ `02b` → `04`)

## Limites

- Échantillon de 1000 produits (échantillon gratuit de The Beauty API, 180 000+ dans la version complète payante) — volume suffisant pour du transfer learning mais limité pour des sous-catégories rares (ex. `fragrance` : 3 produits)
- Style photographique hétérogène (photos professionnelles sur fond blanc mêlées à des photos amateur) — facteur de bruit potentiel non contrôlé
- Les embeddings CLIP, bien que performants, restent peu interprétables directement ; l'explicabilité repose ici sur les features manuelles et l'analyse des coefficients, pas sur une explication image par image
- Résultats obtenus sur un échantillon États-Unis/international du catalogue The Beauty API — la généralisation à un marché spécifique (France, par exemple) n'a pas été testée

## Auteure

Elise Deyris — 4ème année, double diplôme CentraleSupélec × ESSEC (AIDAMS, Data Science, IA et sciences du management)
[GitHub](https://github.com/elisemyr)