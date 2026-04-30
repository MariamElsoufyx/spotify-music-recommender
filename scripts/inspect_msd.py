import h5py

path = "D:/Projects/spotify-music-recommender/data/raw/million_song/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5"

with h5py.File(path, "r") as f:
    s = f["analysis"]["songs"]
    print("All analysis/songs fields:", list(s.dtype.names))
    print()
    print("metadata/songs fields:    ", list(f["metadata"]["songs"].dtype.names))
    print()
    print("musicbrainz/songs fields: ", list(f["musicbrainz"]["songs"].dtype.names))
    print()
    print("=== Sample Values ===")
    print("song_id:     ", f["metadata"]["songs"]["song_id"][0])
    print("title:       ", f["metadata"]["songs"]["title"][0])
    print("artist:      ", f["metadata"]["songs"]["artist_name"][0])
    print("year:        ", f["musicbrainz"]["songs"]["year"][0])
    print("tempo:       ", s["tempo"][0])
    print("loudness:    ", s["loudness"][0])
    print("energy:      ", s["energy"][0])
    print("danceability:", s["danceability"][0])
    print("key:         ", s["key"][0])
    print("mode:        ", s["mode"][0])
