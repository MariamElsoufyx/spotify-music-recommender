"""
Download all Kaggle datasets into data/raw/.
Requirements:
  - pip install kaggle
  - Place your kaggle.json API token at ~/.kaggle/kaggle.json
Run:
  python scripts/download_data.py
"""

import os
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = [
    {
        "slug": "tomigelo/spotify-audio-features",
        "dest": RAW_DIR / "spotify_audio_features",
    },
    {
        "slug": "mrmorj/dataset-of-songs-in-spotify",
        "dest": RAW_DIR / "dataset_of_songs",
    },
    {
        "slug": "maharshipandya/-spotify-tracks-dataset",
        "dest": RAW_DIR / "spotify_tracks",
    },
    {
        "slug": "zaheenhamidani/ultimate-spotify-tracks-db",
        "dest": RAW_DIR / "ultimate_spotify_tracks",
    },
    {
        "slug": "tonygordonjr/spotify-dataset-2023",
        "dest": RAW_DIR / "spotify_2023",
    },
]


def authenticate() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def download_all(api: KaggleApi) -> None:
    for ds in DATASETS:
        dest = ds["dest"]
        dest.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded (folder is non-empty)
        if any(dest.iterdir()):
            print(f"[skip] {ds['slug']} — already exists at {dest}")
            continue

        print(f"[downloading] {ds['slug']} → {dest}")
        owner, name = ds["slug"].split("/", 1)
        api.dataset_download_files(
            dataset=ds["slug"],
            path=str(dest),
            unzip=True,
            quiet=False,
        )
        print(f"[done] {ds['slug']}")


if __name__ == "__main__":
    try:
        api = authenticate()
    except Exception as e:
        print(f"[error] Kaggle authentication failed: {e}")
        print("Make sure ~/.kaggle/kaggle.json exists and is valid.")
        sys.exit(1)

    download_all(api)
    print("\nAll datasets downloaded to data/raw/")
    print("Note: Million Song Dataset must be downloaded manually from http://millionsongdataset.com/")
