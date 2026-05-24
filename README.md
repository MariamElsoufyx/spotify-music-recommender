# Spotify Music Recommender

A hybrid music recommendation system built with PySpark that combines content-based filtering (audio features + cosine similarity) and collaborative filtering (ALS matrix factorization).

---

## Overview

The system processes data from multiple Spotify datasets and the Million Song Dataset to recommend songs through two complementary approaches:

- **Content-Based**: Finds songs acoustically similar to an input playlist using LSH-accelerated cosine similarity across 738K+ songs.
- **Collaborative Filtering**: Predicts songs a user will enjoy based on 11.3M user-playlist interactions using ALS with implicit feedback.

---

## Project Structure

```
spotify-music-recommender/
├── notebooks/               # Main analysis and modeling pipeline
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_1_preprocessing_songs.ipynb    # Song feature preprocessing
│   ├── 02_2_preprocessing_users.ipynb    # User interaction preprocessing
│   ├── 02_3_preprocessing_MSD.ipynb      # Million Song Dataset preprocessing
│   ├── 03_cosine_similarity.ipynb        # Content-based recommendations
│   └── 04_als_model.ipynb               # Collaborative filtering model
├── scripts/
│   ├── download_data.py     # Downloads Kaggle datasets
│   └── create_old_playlist.py
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and normalized data
├── outputs/
│   ├── models/              # Saved ALS model
│   ├── playlists/           # Test playlists
│   └── similarity_matrix/   # Precomputed similarity data
├── docs/
└── requirements.txt
```

---

## Pipeline

Run the notebooks in order:

| Step | Notebook | Output |
|------|----------|--------|
| 1 | `01_eda.ipynb` | Feature analysis and correlation insights |
| 2 | `02_1_preprocessing_songs.ipynb` | 738,904 cleaned songs with 8 audio features |
| 3 | `02_2_preprocessing_users.ipynb` | 11.3M user-song interactions, train/val/test split |
| 4 | `02_3_preprocessing_MSD.ipynb` | 9,851 MSD records with artist popularity features |
| 5 | `03_cosine_similarity.ipynb` | Top-50 content-based recommendations |
| 6 | `04_als_model.ipynb` | Trained ALS model + top-10 personalized recommendations |

---

## Data Sources

Download via `python scripts/download_data.py` (requires Kaggle API credentials):

- Spotify Audio Features (2018, 2019)
- Spotify Tracks Dataset
- Ultimate Spotify Tracks Database
- Spotify 2023 (albums, features, tracks)
- Spotify Playlists (user interaction data)
- [Million Song Subset](http://millionsongdataset.com/) — manual download required (10K HDF5 files)

---

## Models

### Content-Based (Notebook 03)

- Audio features used: `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- Bucketed Random Projection LSH for approximate nearest-neighbor search
- Blends 40 similarity-ranked songs + 10 same-artist songs per recommendation set

### Collaborative Filtering (Notebook 04)

ALS with implicit feedback (PySpark MLlib).

**Best configuration**: `rank=50`, `regParam=0.1`, `alpha=20.0`

| Split | HR@10 | MRR@10 |
|-------|-------|--------|
| Validation | 0.7008 | 0.5195 |
| Test | ~0.64 | ~0.43 |

Evaluated using 99-negative-sample ranking per query.

---

## Setup

**Requirements**: Python 3, Java 8+, Hadoop (Windows), PySpark 3.5+

```bash
pip install -r requirements.txt
```

On Windows, set the Hadoop binary path before starting Spark:

```python
import os
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] += r";C:\hadoop\bin"
```

Then launch Jupyter and run notebooks in order.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Distributed Processing | PySpark 3.5, Hadoop |
| Machine Learning | PySpark MLlib (ALS, LSH, VectorAssembler, MinMaxScaler) |
| Data Analysis | pandas, numpy, scikit-learn |
| Visualization | matplotlib, seaborn |
| Notebooks | Jupyter |
