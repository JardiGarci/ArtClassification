from datasets import load_dataset
import sqlite3

con = sqlite3.connect("data.db")
con.execute("PRAGMA journal_mode=WAL")
con.execute("PRAGMA foreign_keys=ON")



# Only the label columns are requested, so the images are never downloaded.
dataset = load_dataset(
    "huggan/wikiart", split="train", streaming=True,
    columns=["artist", "genre", "style"],
)

# Labels come as class indices; these lists map them back to names.
artists = dataset.features["artist"].names
genres = dataset.features["genre"].names
movements = dataset.features["style"].names

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


# Features

features_names_dfa = ['a_max','a_min','dif_a','a_star','dif_L','dif_R','asy_i','f_max','f_min','dif_f','a','b','c','Hurst']
cols_dfa = ", ".join(f'"{c}" REAL' for c in [f'{name_feat}/seg{grid}{pos+1}' for grid in range(1, 4) for pos in range(grid**2) for name_feat in features_names_dfa])

features_names_renyi = ['a_max','a_min','dif_a','a_star','dif_L','dif_R','asy_i','f_max','f_min','dif_f','D0','D1','D2']
cols_renyi = ", ".join(f'"{c}" REAL' for c in [f'{name_feat}/seg{grid}{pos+1}' for grid in range(1, 4) for pos in range(grid**2) for name_feat in features_names_renyi])

con.executescript(f"""
    DROP TABLE IF EXISTS mfdfa_b1;
    CREATE TABLE mfdfa_b1 (
        image_id INTEGER NOT NULL PRIMARY KEY,
        {cols_dfa}
    );

    DROP TABLE IF EXISTS mfdfa_b2;
        CREATE TABLE mfdfa_b2 (
            image_id INTEGER NOT NULL PRIMARY KEY,
            {cols_dfa}
        );
    
    
    DROP TABLE IF EXISTS mfrenyi_b1;
        CREATE TABLE mfrenyi_b1 (
            image_id INTEGER NOT NULL PRIMARY KEY,
            {cols_renyi}
        );
    
        DROP TABLE IF EXISTS mfrenyi_b2;
            CREATE TABLE mfrenyi_b2 (
                image_id INTEGER NOT NULL PRIMARY KEY,
                {cols_renyi}
            );

""")


con.commit()
con.close()