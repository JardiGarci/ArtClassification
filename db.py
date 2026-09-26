import sqlite3
from datasets import load_dataset

DB_PATH = "data.db"

FEATURES_DFA = ['a_max','a_min','dif_a','a_star','dif_L','dif_R','asy_i',
                'f_max','f_min','dif_f','a','b','c','Hurst']

FEATURES_RENYI = ['a_max','a_min','dif_a','a_star','dif_L','dif_R','asy_i',
                  'f_max','f_min','dif_f','D0','D1','D2']


def connect(db_path=DB_PATH):
    """Open a connection with WAL journaling and foreign key enforcement."""
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def segment_names():
    """The 14 image segments: 1x1, 2x2 and 3x3 partitions."""
    return [f'seg{grid}{pos + 1}'
            for grid in range(1, 4)
            for pos in range(grid ** 2)]


def feature_columns(features, band):
    """Column names for a feature set and a scale band."""
    return [f'{feat}/{seg}/{band}'
            for seg in segment_names()
            for feat in features]


def create_metadata_table(con):
    """Painting catalogue. Only the label columns are requested, so images are never downloaded."""
    dataset = load_dataset(
        "huggan/wikiart", split="train", streaming=True,
        columns=["artist", "genre", "style"],
    )

    # Labels come as class indices; these lists map them back to names.
    artists = dataset.features["artist"].names
    genres = dataset.features["genre"].names
    movements = dataset.features["style"].names   # WikiArt calls the art movement "style"

    con.executescript("""
        DROP TABLE IF EXISTS metadata;
        CREATE TABLE metadata (
            painting_id INTEGER PRIMARY KEY,
            artist      TEXT NOT NULL,
            genre       TEXT NOT NULL,
            movement    TEXT NOT NULL
        );
    """)

    # painting_id is the position of the painting in the dataset, used later to link features.
    rows = ((i, artists[item["artist"]], genres[item["genre"]], movements[item["style"]])
            for i, item in enumerate(dataset))

    con.executemany(
        "INSERT INTO metadata (painting_id, artist, genre, movement) VALUES (?, ?, ?, ?)",
        rows,
    )
    con.commit()

    n = con.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    print(f"Table metadata created with {n} paintings")


def create_feature_table(con, table, features, band, drop=False):
    """Create a feature table with one column per segment and descriptor."""
    if drop:
        con.execute(f'DROP TABLE IF EXISTS "{table}"')

    cols = ",\n            ".join(
        f'"{c}" REAL' for c in feature_columns(features, band)
    )
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table}" (
            painting_id INTEGER NOT NULL PRIMARY KEY
                REFERENCES metadata(painting_id) ON DELETE CASCADE,
            {cols}
        )
    """)
    con.commit()

    n = len(feature_columns(features, band))
    print(f"Table {table} created with {n} feature columns")


def create_mfdfa_b1(con, drop=True):
    """MF-DFA features, band 1 (6 px to 25% of the segment side)."""
    create_feature_table(con, "mfdfa_b1", FEATURES_DFA, "b1", drop)

def create_mfdfa_b2(con, drop=True):
    """MF-DFA features, band 2 (25% to 75% of the segment side)."""
    create_feature_table(con, "mfdfa_b2", FEATURES_DFA, "b2", drop)

def create_mfrenyi_b1(con, drop=True):
    """MF-Rényi features, band 1 (6 px to 25% of the segment side)."""
    create_feature_table(con, "mfrenyi_b1", FEATURES_RENYI, "b1", drop)

def create_mfrenyi_b2(con, drop=True):
    """MF-Rényi features, band 2 (25% to 75% of the segment side)."""
    create_feature_table(con, "mfrenyi_b2", FEATURES_RENYI, "b2", drop)


def insert_features(con, table, painting_id, features, commit=True):
    """Insert or replace one painting's feature row.

    features: dict mapping column name to value, e.g. {'a_max/seg11/b1': 0.83, ...}.
    Missing columns are left as NULL. Values that are NaN are stored as NULL.
    """
    row = {"painting_id": painting_id}
    row.update({name: float(value) for name, value in features.items()})

    cols = ", ".join(f'"{c}"' for c in row)
    placeholders = ", ".join("?" for _ in row)

    con.execute(
        f'INSERT OR REPLACE INTO "{table}" ({cols}) VALUES ({placeholders})',
        list(row.values()),
    )
    if commit:
        con.commit()

# =============================================================================
# Extraction bookkeeping
# =============================================================================

def create_extraction_log(con):
    """Per painting, per method and band: whether extraction finished or failed."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS extraction_log (
            painting_id INTEGER NOT NULL,
            method      TEXT    NOT NULL,
            band        TEXT    NOT NULL,
            status      TEXT    NOT NULL CHECK (status IN ('ok', 'error')),
            error       TEXT,
            updated_at  TEXT    NOT NULL,
            PRIMARY KEY (painting_id, method, band)
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS ix_log_status
        ON extraction_log (method, band, status)
    """)
    con.commit()


def log_extraction(con, painting_id, method, band, status, error=None, commit=True):
    """Record the outcome of one painting. Replaces any previous entry."""
    con.execute(
        "INSERT OR REPLACE INTO extraction_log"
        " (painting_id, method, band, status, error, updated_at)"
        " VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (painting_id, method, band, status,
         None if error is None else str(error)[:500]),
    )
    if commit:
        con.commit()


def done_painting_ids(con, table):
    """Painting ids already present in a feature table."""
    return {row[0] for row in con.execute(f'SELECT painting_id FROM "{table}"')}


def failed_painting_ids(con, method, band):
    """Painting ids whose last attempt ended in error."""
    return {row[0] for row in con.execute(
        "SELECT painting_id FROM extraction_log"
        " WHERE method = ? AND band = ? AND status = 'error'",
        (method, band),
    )}
