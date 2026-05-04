import glob
import pandas as pd

songs_test_files = glob.glob("data/processed/songs_cleaned/*.csv")
songs_test_df = pd.read_csv(songs_test_files[0], on_bad_lines="skip", engine="python")

playlist = (
    songs_test_df[["track_id", "track_name", "artist_name"]]
    .sample(n=20, random_state=42)
    .reset_index(drop=True)
)

playlist.to_csv("data/old_playlist.csv", index=False)

print(f"Saved 20 songs to data/old_playlist.csv")
print(f"Columns: {playlist.columns.tolist()}")
