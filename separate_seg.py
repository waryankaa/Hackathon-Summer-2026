from pathlib import Path

import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path(
    "/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/data/full_train.csv"
)

OUTPUT_DIR = Path(
    "/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/segments"
)


# ============================================================
# SETTINGS
# ============================================================

# A gene must be expressed (> 0) in at least this many cells
# within a segment to be kept.

MIN_NONZERO_CELLS = 5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")


# ============================================================
# CHECK SEGMENT COLUMN
# ============================================================

if "Segment" not in df.columns:

    raise ValueError(
        "The input file does not contain a 'Segment' column."
    )


# ============================================================
# DEFINE NON-GENE COLUMNS
# ============================================================

metadata_columns = [
    "Datasets",
    "volume",
    "center_x",
    "center_y",
    "MERFISH_cell_type_annotation",
    "Region",
    "Excitatory_vs_Inhibitory",
    "Segment",
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID",
]


# ============================================================
# IDENTIFY CELL ID COLUMN
# ============================================================

first_column = df.columns[0]

if (
    str(first_column).startswith("Unnamed:")
    or df[first_column].nunique() == len(df)
):

    if first_column not in metadata_columns:

        metadata_columns.append(first_column)

    print(
        f"Treating '{first_column}' as the cell ID column."
    )


# ============================================================
# IDENTIFY GENE COLUMNS
# ============================================================

gene_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


print(
    f"Number of gene-expression columns: "
    f"{len(gene_columns)}"
)


# ============================================================
# CREATE TEMPORARY SEGMENT VALUE
# ============================================================

df["Segment_for_split"] = (
    df["Segment"]
    .fillna("NA")
    .astype(str)
)


# ============================================================
# SPLIT DATA BY SEGMENT
# ============================================================

for segment, segment_df in df.groupby(
    "Segment_for_split"
):

    segment_df = segment_df.copy()


    # --------------------------------------------------------
    # REMOVE TEMPORARY COLUMN
    # --------------------------------------------------------

    segment_df = segment_df.drop(
        columns=["Segment_for_split"]
    )


    # --------------------------------------------------------
    # FIND GENES WITH FEWER THAN 5 NON-ZERO CELLS
    # --------------------------------------------------------

    genes_in_segment = [
        gene
        for gene in gene_columns
        if gene in segment_df.columns
    ]


    genes_to_remove = []


    for gene in genes_in_segment:

        # Convert values to numeric
        gene_values = pd.to_numeric(
            segment_df[gene],
            errors="coerce"
        ).fillna(0)


        # Count how many cells have expression > 0
        nonzero_count = (
            gene_values > 0
        ).sum()


        # Remove gene if fewer than 5 cells express it
        if nonzero_count < MIN_NONZERO_CELLS:

            genes_to_remove.append(
                gene
            )


    # --------------------------------------------------------
    # REMOVE LOW-FREQUENCY GENES
    # --------------------------------------------------------

    segment_df = segment_df.drop(
        columns=genes_to_remove
    )


    # --------------------------------------------------------
    # CLEAN SEGMENT NAME
    # --------------------------------------------------------

    if segment.endswith(".0"):

        segment = segment[:-2]


    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"segment_{segment}.csv"
    )


    segment_df.to_csv(
        output_file,
        index=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    remaining_genes = (
        len(genes_in_segment)
        - len(genes_to_remove)
    )


    print("\n" + "=" * 60)

    print(
        f"Segment {segment}"
    )

    print(
        f"Cells: "
        f"{len(segment_df):,}"
    )

    print(
        f"Original genes: "
        f"{len(genes_in_segment):,}"
    )

    print(
        f"Genes expressed in fewer than "
        f"{MIN_NONZERO_CELLS} cells removed: "
        f"{len(genes_to_remove):,}"
    )

    print(
        f"Genes remaining: "
        f"{remaining_genes:,}"
    )

    print(
        f"Saved as: "
        f"{output_file.name}"
    )


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)