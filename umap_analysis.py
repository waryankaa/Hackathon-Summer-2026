import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# LOAD DATA
# ============================================================

FILE_PATH = "segments/segment_NA.csv"

df = pd.read_csv(FILE_PATH)

print(f"Number of cells: {len(df)}")


# ============================================================
# METADATA COLUMNS
# ============================================================

metadata_columns = [
    "Unnamed: 0",
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
# FIND GENE COLUMNS
# ============================================================

gene_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]

X = df[gene_columns].copy()

# Make sure everything is numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)

print(f"Starting genes: {len(gene_columns)}")


# ============================================================
# REMOVE VERY SPARSE GENES
# ============================================================

# Keep genes expressed in at least 5% of cells

percent_expressed = (X > 0).mean()

genes_to_keep = percent_expressed[
    percent_expressed >= 0.05
].index

X = X[genes_to_keep]

print(
    f"Genes expressed in >=5% of cells: "
    f"{X.shape[1]}"
)


# ============================================================
# LOG TRANSFORM
# ============================================================

X = np.log1p(X)


# ============================================================
# SELECT MOST VARIABLE GENES
# ============================================================

gene_variance = X.var()

gene_variance = gene_variance.sort_values(
    ascending=False
)

TOP_N_GENES = min(
    50,
    len(gene_variance)
)

top_genes = gene_variance.head(
    TOP_N_GENES
).index

X = X[top_genes]

print(
    f"Using top {len(top_genes)} "
    f"variable genes."
)

print("\nTop genes:")

for gene in top_genes:
    print(gene)


# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# PCA
# ============================================================

# PCA removes some noise before UMAP.

N_PCS = min(
    20,
    X_scaled.shape[1]
)

pca = PCA(
    n_components=N_PCS,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

print(
    "\nVariance explained by PCA:",
    pca.explained_variance_ratio_.sum()
)


# ============================================================
# UMAP
# ============================================================

print("\nRunning UMAP...")

reducer = umap.UMAP(

    # Smaller = focuses more on local neighborhoods
    n_neighbors=10,

    # Smaller = tighter clusters
    min_dist=0.05,

    n_components=2,

    metric="euclidean",

    random_state=42
)

embedding = reducer.fit_transform(X_pca)

print("UMAP finished!")


# ============================================================
# SAVE COORDINATES
# ============================================================

df["UMAP_1"] = embedding[:, 0]
df["UMAP_2"] = embedding[:, 1]


# ============================================================
# PLOT BY CELL TYPE
# ============================================================

plt.figure(
    figsize=(12, 9)
)

cell_types = sorted(
    df[
        "MERFISH_cell_type_annotation"
    ]
    .dropna()
    .unique()
)


for cell_type in cell_types:

    mask = (
        df["MERFISH_cell_type_annotation"]
        == cell_type
    )

    plt.scatter(
        df.loc[mask, "UMAP_1"],
        df.loc[mask, "UMAP_2"],
        s=20,
        alpha=0.7,
        label=cell_type
    )


plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.title(
    "UMAP Using Variable MERFISH Genes"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    "MERFISH_UMAP_variable_genes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()