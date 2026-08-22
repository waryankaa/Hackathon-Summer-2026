import pandas as pd

# Read the data
df = pd.read_csv("/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/segments/segment_NA.csv")


CELL_TYPE_COLUMN = "MERFISH_cell_type_annotation"


# ============================================================
# CELL TYPE FREQUENCY
# ============================================================

cell_type_counts = df[CELL_TYPE_COLUMN].value_counts()

df["cell_type_frequency"] = (
    df[CELL_TYPE_COLUMN].map(cell_type_counts)
    / len(df)
)


# ============================================================
# NORMALIZE FREQUENCY BETWEEN 0 AND 1
# ============================================================

min_freq = df["cell_type_frequency"].min()
max_freq = df["cell_type_frequency"].max()

df["normalized_cell_type_frequency"] = (
    df["cell_type_frequency"] - min_freq
) / (
    max_freq - min_freq
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "segment_NA_with_frequency.csv",
    index=False
)

print("Done!")

print(
    df[
        [
            CELL_TYPE_COLUMN,
            "cell_type_frequency",
            "normalized_cell_type_frequency",
        ]
    ].head()
)