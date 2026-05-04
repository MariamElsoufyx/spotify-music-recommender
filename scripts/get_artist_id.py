"""
Enrich genres_v2.csv with the primary artist (artist_0) name and id
by querying the Spotify Web API.

Reads track IDs from data/raw/dataset_of_songs/genres_v2.csv, fetches
metadata in batches of 50, and writes a mapping CSV to
data/processed/genres_v2_artists.csv with columns:
    track_id, artist_0_id, artist_0_name

Auth: Client Credentials flow. Set SPOTIPY_CLIENT_ID and
SPOTIPY_CLIENT_SECRET in a .env file at the project root, or as
environment variables. Get credentials at
https://developer.spotify.com/dashboard.

Usage:
    python scripts/get_artist_id.py
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "dataset_of_songs" / "genres_v2.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "genres_v2_artists.csv"

TOKEN_URL = "https://accounts.spotify.com/api/token"
TRACKS_URL = "https://api.spotify.com/v1/tracks"
BATCH_SIZE = 50


def get_access_token(client_id: str, client_secret: str) -> str:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        TOKEN_URL,
        headers={"Authorization": f"Basic {auth}"},
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    if not r.ok:
        print(f"[auth-error] HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    payload = r.json()
    token = payload.get("access_token", "")
    print(f"[auth] got token (len={len(token)}, prefix={token[:6]!r}, "
          f"expires_in={payload.get('expires_in')})")
    if not token:
        raise RuntimeError(f"Empty access_token in response: {payload}")
    return token


def fetch_batch(ids: list[str], token: str) -> tuple[list[dict], str]:
    """Fetch up to 50 tracks. Returns (tracks, possibly-refreshed-token)."""
    while True:
        r = requests.get(
            TRACKS_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={"ids": ",".join(ids)},
            timeout=20,
        )
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "1"))
            print(f"[rate-limit] sleeping {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            # token expired mid-run
            client_id = os.environ["SPOTIPY_CLIENT_ID"]
            client_secret = os.environ["SPOTIPY_CLIENT_SECRET"]
            token = get_access_token(client_id, client_secret)
            continue
        if not r.ok:
            print(f"\n[error] HTTP {r.status_code} from Spotify")
            print(f"[error] response body: {r.text[:500]}")
            print(f"[error] first id in batch: {ids[0]}")
        r.raise_for_status()
        return r.json().get("tracks", []), token


def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=["track_id", "artist_0_id", "artist_0_name"])


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("[error] Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET "
              "in a .env file or environment.")
        sys.exit(1)

    if not INPUT_CSV.exists():
        print(f"[error] Input not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, usecols=["id"], dtype=str)
    all_ids = df["id"].dropna().drop_duplicates().tolist()
    print(f"[info] {len(all_ids)} unique track IDs in {INPUT_CSV.name}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(OUTPUT_CSV)
    done = set(existing["track_id"]) if not existing.empty else set()
    todo = [tid for tid in all_ids if tid not in done]
    if not todo:
        print(f"[done] All IDs already in {OUTPUT_CSV.name}")
        return
    print(f"[info] {len(todo)} IDs to fetch ({len(done)} already cached)")

    token = get_access_token(client_id, client_secret)

    write_header = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", encoding="utf-8", newline="") as fh:
        if write_header:
            fh.write("track_id,artist_0_id,artist_0_name\n")

        for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="batches"):
            batch = todo[i : i + BATCH_SIZE]
            tracks, token = fetch_batch(batch, token)

            # The API returns tracks in request order; nulls for invalid IDs.
            for tid, track in zip(batch, tracks):
                if not track or not track.get("artists"):
                    fh.write(f"{tid},,\n")
                    continue
                a0 = track["artists"][0]
                name = (a0.get("name") or "").replace('"', '""')
                fh.write(f'{tid},{a0.get("id", "")},"{name}"\n')
            fh.flush()

    print(f"[done] Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
