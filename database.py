from datasets import load_dataset
import sqlite3

# Only the label columns are requested, so the images are never downloaded.
dataset = load_dataset(
    "huggan/wikiart", split="train", streaming=True,
    columns=["artist", "genre", "style"],
)

# Labels come as class indices; these lists map them back to names.
artists = dataset.features["artist"].names
genres = dataset.features["genre"].names
movements = dataset.features["style"].names

con = sqlite3.connect("data.db")
con.execute("PRAGMA journal_mode=WAL")
con.execute("PRAGMA foreign_keys=ON")

# The table is rebuilt on every run.
# painting_id is the position of the painting in the dataset, used later to link features.
con.executescript("""
DROP TABLE IF EXISTS metadata;
CREATE TABLE metadata (
    painting_id  INTEGER PRIMARY KEY,
    artist       TEXT NOT NULL,
    genre        TEXT NOT NULL,
    movement     TEXT NOT NULL
);
""")

# WikiArt calls the art movement "style".
for i, item in enumerate(dataset):
    artist = artists[item["artist"]]
    genre = genres[item["genre"]]
    movement = movements[item["style"]]
    con.execute(
        "INSERT INTO metadata (painting_id, artist, genre, movement) VALUES (?, ?, ?, ?)",
        (i, artist, genre, movement),
    )

con.commit()
con.close()