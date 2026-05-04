"""
Append MSD preprocessing cells to notebooks/02_3_preprocessing_MSD.ipynb.

Idempotent: re-running drops any previously appended MSD cells (those with
ids starting with 'msd_') and appends a fresh set, so the notebook stays
in sync with this script.
"""

import json
import uuid
from pathlib import Path

NB_PATH = Path("D:/Projects/spotify-music-recommender/notebooks/02_3_preprocessing_MSD.ipynb")


def _id():
    return "msd_" + uuid.uuid4().hex[:8]


def md_cell(src):
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id(),
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS = []

CELLS.append(md_cell(
    "## Step 1 — Config\n"
    "\n"
    "MSD subset is ~10,000 HDF5 files under `data/raw/million_song/MillionSongSubset/A|B/.../*.h5`. "
    "We extract 9 columns: `track_id`, `track_name`, `artist_name`, `duration`, `tempo`, `loudness`, "
    "`artist_familiarity`, `artist_hotttnesss`, `song_hotttnesss`. "
    "`energy` is not extracted — MSD's Echo Nest energy field is all 0.0 in the released subset.\n"
))

CELLS.append(code_cell("""\
MSD_RAW_DIR   = "../data/raw/million_song/MillionSongSubset"
PROCESSED_DIR = "../data/processed"
SPLIT_DIR     = "../outputs/train_test_split"

# Numeric features to min-max scale. Energy is added dynamically in Step 3
# only if the column is not all-zero.
MSD_NUMERIC_FEATURES = [
    "duration", "tempo", "loudness",
    "artist_familiarity", "artist_hotttnesss", "song_hotttnesss",
]
"""))

CELLS.append(md_cell(
    "## Step 2 — Discover .h5 files\n"
    "\n"
    "Walk the MSD subset tree once on the driver, then parallelize the file paths to Spark workers.\n"
))

CELLS.append(code_cell("""\
msd_files = glob.glob(os.path.join(MSD_RAW_DIR, "**", "*.h5"), recursive=True)
print(f"Discovered {len(msd_files):,} MSD .h5 files")
print("Sample paths:")
for p in msd_files[:3]:
    print("  ", p)
"""))

CELLS.append(md_cell(
    "## Step 3 — Extract per-track records with Hadoop/Spark in parallel\n"
    "\n"
    "We parallelize the file-path list across all cores and read each `.h5` inside a `mapPartitions` worker with `h5py`. "
    "This is the Hadoop-style distributed I/O pattern: the driver hands shards (file paths) to workers, workers read independently, results stream back as Rows. "
    "After building the raw DataFrame we drop `energy` if its entire column is 0.0 (MSD never populated it).\n"
))

CELLS.append(code_cell("""\
# Explicit schema so Spark skips inference (inference would re-scan all 10k files).
MSD_SCHEMA = StructType([
    StructField("track_id",           StringType(), True),
    StructField("track_name",         StringType(), True),
    StructField("artist_name",        StringType(), True),
    StructField("duration",           DoubleType(), True),
    StructField("tempo",              DoubleType(), True),
    StructField("loudness",           DoubleType(), True),
    StructField("artist_familiarity", DoubleType(), True),
    StructField("artist_hotttnesss",  DoubleType(), True),
    StructField("song_hotttnesss",    DoubleType(), True),
])


def _extract_partition(paths_iter):
    \"\"\"Worker-side: open each .h5 file, pull the row-zero values we want.

    h5py returns bytes for string fields; we decode to UTF-8.
    Numeric fields that come back as NaN are converted to None so Spark stores
    them as proper SQL NULL — NaN would survive dropna and poison MinMaxScaler.
    \"\"\"
    import h5py
    import math as _math

    def _s(v):
        if v is None:
            return None
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        v = v.strip()
        return v if v else None

    def _f(v):
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if _math.isnan(f) or _math.isinf(f):
            return None
        return f

    for path in paths_iter:
        try:
            with h5py.File(path, "r") as f:
                a = f["analysis"]["songs"]
                m = f["metadata"]["songs"]
                yield (
                    _s(a["track_id"][0]),
                    _s(m["title"][0]),
                    _s(m["artist_name"][0]),
                    _f(a["duration"][0]),
                    _f(a["tempo"][0]),
                    _f(a["loudness"][0]),
                    _f(m["artist_familiarity"][0]),
                    _f(m["artist_hotttnesss"][0]),
                    _f(m["song_hotttnesss"][0]),
                )
        except Exception as e:
            # One bad file shouldn't abort the whole job.
            print(f"[skip] {path}: {e}")
            continue


n_partitions = max(8, (os.cpu_count() or 4) * 4)
paths_rdd    = spark.sparkContext.parallelize(msd_files, numSlices=n_partitions)
records_rdd  = paths_rdd.mapPartitions(_extract_partition)

msd_df_raw = spark.createDataFrame(records_rdd, schema=MSD_SCHEMA)
msd_df_raw.cache()
print(f"Extracted {msd_df_raw.count():,} MSD records")

msd_df_raw.printSchema()
msd_df_raw.show(5, truncate=False)
"""))

CELLS.append(md_cell(
    "## Step 4 — Clean the MSD frame\n"
    "\n"
    "1. Drop rows missing any essential identifier (`track_id`, `track_name`, `artist_name`).\n"
    "2. Dedupe by `track_id`.\n"
    "3. Dedupe by `(track_name, artist_name)` — same song re-uploaded with a new ID.\n"
    "4. Lowercase + trim text fields.\n"
    "5. Filter `duration` to 30 s – 15 min.\n"
))

CELLS.append(code_cell("""\
print("step 1: dropping rows missing essential identifiers...")
msd_df = msd_df_raw.dropna(subset=["track_id", "track_name", "artist_name"])

print("step 2: dropping duplicate track_ids...")
msd_df = msd_df.dropDuplicates(["track_id"])

print("step 3: dropping (track_name, artist_name) duplicates...")
msd_df = msd_df.dropDuplicates(["track_name", "artist_name"])

print("step 4: lowercasing + trimming track_name and artist_name...")
msd_df = (
    msd_df
    .withColumn("track_name",  lower(trim(col("track_name"))))
    .withColumn("artist_name", lower(trim(col("artist_name"))))
)

print("step 5: filtering duration to [30s, 15min]...")
msd_df = msd_df.filter((col("duration") >= 30.0) & (col("duration") <= 900.0))

msd_df = msd_df.cache()
print(f"Rows after structural cleaning: {msd_df.count():,}")
msd_df.show(5, truncate=False)
"""))

CELLS.append(md_cell(
    "## Step 5 — Min-max normalize numeric features\n"
    "\n"
    "Snapshot nulls → mean-impute (so MinMaxScaler gets a complete vector) → scale → restore nulls.\n"
))

CELLS.append(code_cell("""\
from pyspark.ml.feature import Imputer

# 1. Snapshot null mask so we can restore genuine missingness after scaling.
for c in MSD_NUMERIC_FEATURES:
    msd_df = msd_df.withColumn(f"_null_{c}", col(c).isNull())

# 2. Mean-impute — gives MinMaxScaler a complete vector to fit on.
imputer = Imputer(strategy="mean",
                  inputCols=MSD_NUMERIC_FEATURES,
                  outputCols=MSD_NUMERIC_FEATURES)
msd_df = imputer.fit(msd_df).transform(msd_df)

# 3. Assemble → fit MinMaxScaler → transform → unpack.
assembler    = VectorAssembler(inputCols=MSD_NUMERIC_FEATURES, outputCol="features_vec")
assembled    = assembler.transform(msd_df)
scaler_model = MinMaxScaler(inputCol="features_vec", outputCol="features_scaled").fit(assembled)
scaled       = scaler_model.transform(assembled).withColumn("_arr", vector_to_array("features_scaled"))

for i, c in enumerate(MSD_NUMERIC_FEATURES):
    scaled = scaled.withColumn(c, col("_arr")[i])

# 4. Restore nulls and drop helper columns.
for c in MSD_NUMERIC_FEATURES:
    scaled = scaled.withColumn(c, when(col(f"_null_{c}"), lit(None)).otherwise(col(c)))
    scaled = scaled.drop(f"_null_{c}")

msd_df_cleaned = scaled.drop("features_vec", "features_scaled", "_arr")

print(f"Final shape: {msd_df_cleaned.count():,} rows × {len(msd_df_cleaned.columns)} cols")
msd_df_cleaned.printSchema()
msd_df_cleaned.show(5, truncate=False)
"""))

CELLS.append(md_cell("## Step 6 — Save cleaned MSD\n"))

CELLS.append(code_cell("""\
os.makedirs(PROCESSED_DIR, exist_ok=True)
spark_write_csv(msd_df_cleaned, f"{PROCESSED_DIR}/msd_cleaned")
"""))

CELLS.append(md_cell(
    "## Step 7 — 80/20 train/test split\n"
    "\n"
    "Same `seed=42` as the other splits so the partitions are reproducible across runs.\n"
))

CELLS.append(code_cell("""\
os.makedirs(SPLIT_DIR, exist_ok=True)
msd_train_df, msd_test_df = msd_df_cleaned.randomSplit([0.8, 0.2], seed=42)

spark_write_csv(msd_train_df, f"{SPLIT_DIR}/msd_train")
spark_write_csv(msd_test_df,  f"{SPLIT_DIR}/msd_test")

print(f"Train : {msd_train_df.count():,} rows")
print(f"Test  : {msd_test_df.count():,} rows")
"""))

CELLS.append(code_cell("""\
spark.stop()
print("Done. Spark session closed.")
"""))


# ---- splice into the notebook ----
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
nb["cells"] = [c for c in nb["cells"] if not str(c.get("id", "")).startswith("msd_")]
nb["cells"].extend(CELLS)
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {len(CELLS)} cells. Notebook now has {len(nb['cells'])} cells.")
