"""
Utiliser pour télécharger le dataset thebeautyapi/beautyproducts depuis Hugging Face :
- beauty_data.csv (métadonnées)
- images/*.jpeg (toutes les images)

Commande pour lancer ce file avec le env UV: uv run scripts/download_hf_dataset.py
"""

import requests
import os
import time

BASE_URL = "https://huggingface.co/datasets/thebeautyapi/beautyproducts/resolve/main"
RAW_DIR = "data/raw"
IMAGES_DIR = "data/images"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def download_metadata():
    """Télécharge beauty_data.csv"""
    path = os.path.join(RAW_DIR, "beauty_data.csv")
    if os.path.exists(path):
        print("beauty_data.csv déjà présent, skip.")
        return
    print("Téléchargement de beauty_data.csv...")
    r = requests.get(f"{BASE_URL}/beauty_data.csv", timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"OK — {len(r.content)} octets")


def get_image_list():
    """Liste tous les fichiers images/ via l'API HF"""
    url = "https://huggingface.co/api/datasets/thebeautyapi/beautyproducts/tree/main/images"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    files = r.json()
    return [f["path"] for f in files if f["type"] == "file"]


def download_images():
    """Télécharge chaque image listée, avec reprise si déjà présente"""
    print("Récupération de la liste des images...")
    image_paths = get_image_list()
    print(f"{len(image_paths)} images trouvées dans le repo.")

    success, skipped, failed = 0, 0, 0
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        local_path = os.path.join(IMAGES_DIR, filename)

        if os.path.exists(local_path):
            skipped += 1
            continue

        url = f"{BASE_URL}/{path}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                success += 1
            else:
                print(f"Échec ({r.status_code}) : {filename}")
                failed += 1
        except Exception as e:
            print(f"Erreur {filename} : {e}")
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(image_paths)} traités — succès: {success}, skip: {skipped}, échecs: {failed}")

        time.sleep(0.1)

    print(f"\nTerminé. Succès: {success}, déjà présents: {skipped}, échecs: {failed}")


if __name__ == "__main__":
    download_metadata()
    download_images()