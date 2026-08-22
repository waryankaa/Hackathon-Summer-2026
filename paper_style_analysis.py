from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path(
    "segments/segment_NA.csv"
)

TARGET = "MERFISH_cell_type_annotation"

SECTION_COL = "Section_ID"

X_COL = "center_x"

Y_COL = "center_y"

VOLUME_COL = "volume"


# Fraction of labeled cells that we temporarily hide
# and pretend are the test dataset.

TEST_SIZE = 0.20


# Repeat the experiment several times so that we do not
# trust one lucky/random train-test split.

N_REPEATS = 10


RANDOM_STATE = 42


# Spatial neighborhoods.
#
# These assume center_x and center_y are in micrometers.
#
# We will calculate the cell-type composition around every
# cell at each of these distances.

RADII = [
    25,
    50,
    100,
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print(
    "\n=========================================="
)

print(
    "CELL TYPE CLASSIFICATION TEST"
)

print(
    "=========================================="
)


print(
    f"\nTotal cells: {len(df)}"
)


print(
    "\nCell types:"
)


print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# REMOVE CELLS WITHOUT LABELS
# ============================================================

# Since this program is testing model performance,
# every cell used here needs to have a known true label.

df = df[
    df[TARGET].notna()
].copy()


df = df.reset_index(
    drop=True
)


# ============================================================
# DEFINE METADATA COLUMNS
# ============================================================

# These should NOT accidentally be interpreted as genes.

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
# DETECT GENE COLUMNS
# ============================================================

gene_columns = [

    column

    for column in df.columns

    if column not in metadata_columns
]


# ============================================================
# MAKE SURE GENE COLUMNS ARE NUMERIC
# ============================================================

valid_gene_columns = []


for gene in gene_columns:

    df[gene] = pd.to_numeric(
        df[gene],
        errors="coerce"
    )


    if df[gene].notna().any():

        valid_gene_columns.append(
            gene
        )


gene_columns = valid_gene_columns


# Missing gene counts become zero.

df[gene_columns] = (

    df[gene_columns]

    .fillna(0)
)


# ============================================================
# REMOVE CONSTANT GENES
# ============================================================

# A gene that has exactly the same value in every cell
# cannot help classify cell type.

gene_columns = [

    gene

    for gene in gene_columns

    if df[gene].nunique() > 1
]


print(
    f"\nGene columns being used: "
    f"{len(gene_columns)}"
)


# ============================================================
# LOG TRANSFORM GENE EXPRESSION
# ============================================================

# MERFISH molecule counts can be very skewed.
#
# log1p means:
#
#     log(1 + expression)
#
# Examples:
#
# 0  -> 0
# 1  -> 0.69
# 5  -> 1.79
# 20 -> 3.04
#
# This prevents extremely large counts from dominating
# the classifier.

gene_data = np.log1p(

    df[
        gene_columns
    ].astype(float)
)


gene_data = pd.DataFrame(

    gene_data,

    columns=gene_columns,

    index=df.index
)


# ============================================================
# ALL CELL TYPES
# ============================================================

cell_types = sorted(

    df[TARGET]
    .astype(str)
    .unique()
)


print(
    f"\nNumber of cell types: "
    f"{len(cell_types)}"
)


# ============================================================
# SPATIAL NEIGHBOR FEATURE FUNCTION
# ============================================================

def make_spatial_features(

    query_df,

    reference_df,

    reference_labels,

    cell_types,

    radii,

    query_is_reference=False,
):

    """
    Create spatial-neighborhood features.

    For every QUERY cell:

        Look at labeled REFERENCE cells around it.

    For every radius we calculate:

        number of labeled neighbors

        fraction of neighbors that are:
            astrocyte_1
            astrocyte_2
            microglia
            etc.

    IMPORTANT:

    Validation cells are NEVER used as labeled reference cells.

    Therefore the true label of a validation cell cannot leak
    into its spatial features.
    """


    features = pd.DataFrame(

        index=query_df.index
    )


    # ========================================================
    # INITIALIZE ALL FEATURE COLUMNS
    # ========================================================

    for radius in radii:

        features[
            f"neighbor_count_{radius}"
        ] = 0.0


        for cell_type in cell_types:

            features[
                f"neighbor_{cell_type}_{radius}"
            ] = 0.0


    # ========================================================
    # PROCESS EACH SECTION SEPARATELY
    # ========================================================

    for section in query_df[
        SECTION_COL
    ].dropna().unique():


        query_mask = (

            query_df[
                SECTION_COL
            ]

            == section
        )


        reference_mask = (

            reference_df[
                SECTION_COL
            ]

            == section
        )


        query_section = (

            query_df.loc[
                query_mask
            ]
        )


        reference_section = (

            reference_df.loc[
                reference_mask
            ]
        )


        # ====================================================
        # NO LABELED REFERENCE CELLS
        # ====================================================

        if len(
            reference_section
        ) == 0:

            continue


        # ====================================================
        # GET COORDINATES
        # ====================================================

        ref_coords = (

            reference_section[
                [
                    X_COL,
                    Y_COL,
                ]
            ]

            .apply(
                pd.to_numeric,
                errors="coerce"
            )

            .fillna(0)

            .to_numpy(
                dtype=float
            )
        )


        query_coords = (

            query_section[
                [
                    X_COL,
                    Y_COL,
                ]
            ]

            .apply(
                pd.to_numeric,
                errors="coerce"
            )

            .fillna(0)

            .to_numpy(
                dtype=float
            )
        )


        # ====================================================
        # KD TREE
        # ====================================================

        # This makes spatial neighbor searching much faster.

        tree = cKDTree(
            ref_coords
        )


        ref_labels_section = (

            reference_labels.loc[
                reference_section.index
            ]

            .astype(str)

            .to_numpy()
        )


        # ====================================================
        # CALCULATE NEIGHBORS AT EACH RADIUS
        # ====================================================

        for radius in radii:


            neighbor_lists = (

                tree.query_ball_point(

                    query_coords,

                    r=radius
                )
            )


            # =================================================
            # EACH QUERY CELL
            # =================================================

            for query_position, neighbors in enumerate(
                neighbor_lists
            ):


                query_index = (

                    query_section.index[
                        query_position
                    ]
                )


                # =============================================
                # REMOVE THE CELL ITSELF
                # =============================================

                # When creating features for training cells,
                # the reference set contains the query cell.
                #
                # We must remove it so a cell cannot see
                # its own true label.

                if query_is_reference:


                    filtered_neighbors = []


                    for neighbor_position in neighbors:


                        ref_index = (

                            reference_section.index[
                                neighbor_position
                            ]
                        )


                        if ref_index != query_index:

                            filtered_neighbors.append(
                                neighbor_position
                            )


                    neighbors = filtered_neighbors


                # =============================================
                # NUMBER OF NEIGHBORS
                # =============================================

                n_neighbors = len(
                    neighbors
                )


                features.loc[

                    query_index,

                    f"neighbor_count_{radius}"

                ] = n_neighbors


                # =============================================
                # NO NEIGHBORS
                # =============================================

                if n_neighbors == 0:

                    continue


                # =============================================
                # NEIGHBOR CELL TYPES
                # =============================================

                neighbor_labels = (

                    ref_labels_section[
                        neighbors
                    ]
                )


                # =============================================
                # FRACTION OF EACH CELL TYPE
                # =============================================

                for cell_type in cell_types:


                    fraction = np.mean(

                        neighbor_labels

                        == str(
                            cell_type
                        )
                    )


                    features.loc[

                        query_index,

                        f"neighbor_{cell_type}_{radius}"

                    ] = fraction


    return features.fillna(0)


# ============================================================
# INTERMIXED TRAIN / TEST SPLIT
# ============================================================

def make_intermixed_split(

    df,

    test_size,

    random_state,
):

    """
    Hide approximately 20% of cells WITHIN EACH SECTION.

    This is designed to imitate a hidden test dataset that
    appears spatially intermixed with the training dataset.
    """


    train_indices = []

    test_indices = []


    rng = np.random.default_rng(
        random_state
    )


    # ========================================================
    # EACH SECTION
    # ========================================================

    for section, section_df in df.groupby(
        SECTION_COL
    ):


        indices = (

            section_df.index
            .to_numpy()
        )


        # ----------------------------------------------------
        # Tiny section
        # ----------------------------------------------------

        if len(indices) < 2:

            train_indices.extend(
                indices
            )

            continue


        labels = (

            section_df[
                TARGET
            ]
        )


        # ----------------------------------------------------
        # DETERMINE WHETHER STRATIFICATION IS POSSIBLE
        # ----------------------------------------------------

        value_counts = (

            labels
            .value_counts()
        )


        estimated_test_size = int(

            np.ceil(

                len(section_df)

                * test_size
            )
        )


        can_stratify = (

            len(
                value_counts
            ) > 1

            and

            value_counts.min() >= 2

            and

            estimated_test_size
            >= len(value_counts)
        )


        # ----------------------------------------------------
        # SPLIT
        # ----------------------------------------------------

        try:


            train_idx, test_idx = train_test_split(

                indices,

                test_size=test_size,

                random_state=random_state,

                stratify=(

                    labels

                    if can_stratify

                    else None
                ),
            )


        except ValueError:


            # -----------------------------------------------
            # FALLBACK FOR SMALL SECTIONS
            # -----------------------------------------------

            shuffled = (

                indices.copy()
            )


            rng.shuffle(
                shuffled
            )


            n_test = max(

                1,

                int(

                    round(

                        len(shuffled)

                        * test_size
                    )
                )
            )


            # Make sure at least one training cell remains.

            if n_test >= len(shuffled):

                n_test = (
                    len(shuffled)
                    - 1
                )


            test_idx = (

                shuffled[
                    :n_test
                ]
            )


            train_idx = (

                shuffled[
                    n_test:
                ]
            )


        train_indices.extend(
            train_idx
        )


        test_indices.extend(
            test_idx
        )


    return (

        np.array(
            train_indices
        ),

        np.array(
            test_indices
        )
    )


# ============================================================
# CLASSIFIER
# ============================================================

def build_model(
    random_state
):


    model = ExtraTreesClassifier(

        # Lots of trees gives stable results.
        n_estimators=750,

        # Helps compensate for uneven cell-type frequencies.
        class_weight="balanced",

        # Random subset of features for each split.
        max_features="sqrt",

        min_samples_leaf=1,

        random_state=random_state,

        n_jobs=-1,
    )


    return model


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

results = []


per_class_results = []


# Add confusion matrices across all validation repeats.

confusion_totals = {}


# ============================================================
# REPEATED INTERMIXED VALIDATION
# ============================================================

for repeat in range(
    N_REPEATS
):


    print(
        "\n=========================================="
    )


    print(
        f"Repeat "
        f"{repeat + 1}/"
        f"{N_REPEATS}"
    )


    print(
        "=========================================="
    )


    seed = (

        RANDOM_STATE

        + repeat
    )


    # ========================================================
    # CREATE INTERMIXED TRAIN / VALIDATION SPLIT
    # ========================================================

    train_idx, test_idx = (

        make_intermixed_split(

            df,

            TEST_SIZE,

            seed
        )
    )


    train_df = (

        df.loc[
            train_idx
        ]

        .copy()
    )


    test_df = (

        df.loc[
            test_idx
        ]

        .copy()
    )


    y_train = (

        train_df[
            TARGET
        ]

        .astype(str)
    )


    y_test = (

        test_df[
            TARGET
        ]

        .astype(str)
    )


    print(
        f"\nTraining cells: "
        f"{len(train_df)}"
    )


    print(
        f"Hidden validation cells: "
        f"{len(test_df)}"
    )


    # ========================================================
    # BUILD SPATIAL FEATURES
    # ========================================================

    print(
        "\nBuilding spatial neighborhood features..."
    )


    # --------------------------------------------------------
    # TRAINING CELLS
    # --------------------------------------------------------

    # Training cells can use OTHER training cells as
    # spatial references.

    spatial_train = (

        make_spatial_features(

            query_df=train_df,

            reference_df=train_df,

            reference_labels=y_train,

            cell_types=cell_types,

            radii=RADII,

            query_is_reference=True,
        )
    )


    # --------------------------------------------------------
    # VALIDATION CELLS
    # --------------------------------------------------------

    # Validation cells can ONLY use known training labels.
    #
    # Their true labels remain hidden.

    spatial_test = (

        make_spatial_features(

            query_df=test_df,

            reference_df=train_df,

            reference_labels=y_train,

            cell_types=cell_types,

            radii=RADII,

            query_is_reference=False,
        )
    )


    # ========================================================
    # GENE FEATURES
    # ========================================================

    X_gene_train = (

        gene_data.loc[
            train_idx
        ]

        .copy()
    )


    X_gene_test = (

        gene_data.loc[
            test_idx
        ]

        .copy()
    )


    # ========================================================
    # VOLUME FEATURES
    # ========================================================

    # IMPORTANT:
    #
    # Calculate missing-value replacement using TRAINING data,
    # not the full dataset.

    train_volume = pd.to_numeric(

        train_df[
            VOLUME_COL
        ],

        errors="coerce"
    )


    volume_median = (
        train_volume.median()
    )


    volume_train = (

        train_volume

        .fillna(
            volume_median
        )

        .to_frame(
            name="volume"
        )
    )


    volume_test = (

        pd.to_numeric(

            test_df[
                VOLUME_COL
            ],

            errors="coerce"
        )

        .fillna(
            volume_median
        )

        .to_frame(
            name="volume"
        )
    )


    # ========================================================
    # RAW COORDINATES
    # ========================================================

    coordinate_train = (

        train_df[
            [
                X_COL,
                Y_COL,
            ]
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )

        .fillna(0)
    )


    coordinate_test = (

        test_df[
            [
                X_COL,
                Y_COL,
            ]
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )

        .fillna(0)
    )


    # ========================================================
    # FEATURE SETS
    # ========================================================

    feature_sets = {}


    # --------------------------------------------------------
    # MODEL 1
    #
    # GENES ONLY
    # --------------------------------------------------------

    feature_sets[
        "genes_only"
    ] = (

        X_gene_train,

        X_gene_test
    )


    # --------------------------------------------------------
    # MODEL 2
    #
    # GENES + VOLUME
    # --------------------------------------------------------

    feature_sets[
        "genes_volume"
    ] = (


        pd.concat(

            [
                X_gene_train,
                volume_train,
            ],

            axis=1
        ),


        pd.concat(

            [
                X_gene_test,
                volume_test,
            ],

            axis=1
        ),
    )


    # --------------------------------------------------------
    # MODEL 3
    #
    # GENES + VOLUME + RAW X/Y COORDINATES
    # --------------------------------------------------------

    feature_sets[
        "genes_volume_coordinates"
    ] = (


        pd.concat(

            [
                X_gene_train,
                volume_train,
                coordinate_train,
            ],

            axis=1
        ),


        pd.concat(

            [
                X_gene_test,
                volume_test,
                coordinate_test,
            ],

            axis=1
        ),
    )


    # --------------------------------------------------------
    # MODEL 4
    #
    # GENES + VOLUME + SPATIAL NEIGHBORS
    # --------------------------------------------------------

    feature_sets[
        "genes_volume_neighbors"
    ] = (


        pd.concat(

            [
                X_gene_train,
                volume_train,
                spatial_train,
            ],

            axis=1
        ),


        pd.concat(

            [
                X_gene_test,
                volume_test,
                spatial_test,
            ],

            axis=1
        ),
    )


    # ========================================================
    # TEST EACH MODEL
    # ========================================================

    for model_name, (
        X_train,
        X_test,
    ) in feature_sets.items():


        print(
            f"\nTesting: "
            f"{model_name}"
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = build_model(
            seed
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(

            X_train,

            y_train
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predictions = (

            model.predict(
                X_test
            )
        )


        # ====================================================
        # OVERALL PERFORMANCE
        # ====================================================

        accuracy = accuracy_score(

            y_test,

            predictions
        )


        balanced_accuracy = (

            balanced_accuracy_score(

                y_test,

                predictions
            )
        )


        macro_f1 = f1_score(

            y_test,

            predictions,

            average="macro",

            zero_division=0
        )


        results.append(

            {
                "repeat":
                    repeat + 1,

                "model":
                    model_name,

                "accuracy":
                    accuracy,

                "balanced_accuracy":
                    balanced_accuracy,

                "macro_f1":
                    macro_f1,

                "training_cells":
                    len(train_df),

                "test_cells":
                    len(test_df),
            }
        )


        print(
            f"  Accuracy:          "
            f"{accuracy:.4f}"
        )


        print(
            f"  Balanced accuracy: "
            f"{balanced_accuracy:.4f}"
        )


        print(
            f"  Macro F1:          "
            f"{macro_f1:.4f}"
        )


        # ====================================================
        # PER-CELL-TYPE PERFORMANCE
        # ====================================================

        report = classification_report(

            y_test,

            predictions,

            labels=cell_types,

            output_dict=True,

            zero_division=0
        )


        for cell_type in cell_types:


            cell_result = (

                report.get(

                    cell_type,

                    {}
                )
            )


            per_class_results.append(

                {
                    "repeat":
                        repeat + 1,

                    "model":
                        model_name,

                    "cell_type":
                        cell_type,

                    "precision":
                        cell_result.get(
                            "precision",
                            np.nan
                        ),

                    "recall":
                        cell_result.get(
                            "recall",
                            np.nan
                        ),

                    "f1":
                        cell_result.get(
                            "f1-score",
                            np.nan
                        ),

                    "support":
                        cell_result.get(
                            "support",
                            0
                        ),
                }
            )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(

            y_test,

            predictions,

            labels=cell_types
        )


        if model_name not in confusion_totals:


            confusion_totals[
                model_name
            ] = np.zeros_like(

                cm,

                dtype=float
            )


        confusion_totals[
            model_name
        ] += cm


# ============================================================
# SAVE OVERALL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(

    "classification_model_comparison_all_runs.csv",

    index=False
)


# ============================================================
# OVERALL MODEL SUMMARY
# ============================================================

summary = (

    results_df

    .groupby(
        "model"
    )

    .agg(


        accuracy_mean=(

            "accuracy",

            "mean"
        ),


        accuracy_std=(

            "accuracy",

            "std"
        ),


        balanced_accuracy_mean=(

            "balanced_accuracy",

            "mean"
        ),


        balanced_accuracy_std=(

            "balanced_accuracy",

            "std"
        ),


        macro_f1_mean=(

            "macro_f1",

            "mean"
        ),


        macro_f1_std=(

            "macro_f1",

            "std"
        ),
    )


    .sort_values(

        "macro_f1_mean",

        ascending=False
    )
)


summary.to_csv(

    "classification_model_comparison_summary.csv"
)


print(
    "\n\n=========================================="
)


print(
    "FINAL MODEL COMPARISON"
)


print(
    "==========================================\n"
)


print(
    summary.round(4)
)


print(
    "\nBest model by mean Macro F1:"
)


best_model = (

    summary.index[
        0
    ]
)


print(
    best_model
)


# ============================================================
# PER-CELL-TYPE RESULTS
# ============================================================

per_class_df = pd.DataFrame(

    per_class_results
)


per_class_df.to_csv(

    "per_cell_type_performance_all_runs.csv",

    index=False
)


# ============================================================
# AVERAGE PER-CELL-TYPE PERFORMANCE
# ============================================================

per_class_summary = (

    per_class_df

    .groupby(

        [
            "model",
            "cell_type",
        ]
    )

    .agg(


        precision_mean=(

            "precision",

            "mean"
        ),


        precision_std=(

            "precision",

            "std"
        ),


        recall_mean=(

            "recall",

            "mean"
        ),


        recall_std=(

            "recall",

            "std"
        ),


        f1_mean=(

            "f1",

            "mean"
        ),


        f1_std=(

            "f1",

            "std"
        ),


        mean_support=(

            "support",

            "mean"
        ),
    )


    .reset_index()
)


per_class_summary.to_csv(

    "per_cell_type_performance_summary.csv",

    index=False
)


# ============================================================
# CREATE F1 TABLE
# ============================================================

f1_table = per_class_summary.pivot(

    index="cell_type",

    columns="model",

    values="f1_mean"
)


f1_table.to_csv(

    "per_cell_type_f1_comparison.csv"
)


print(
    "\n\n=========================================="
)


print(
    "PER-CELL-TYPE F1 SCORES"
)


print(
    "==========================================\n"
)


print(
    f1_table.round(3)
)


# ============================================================
# F1 IMPROVEMENT RELATIVE TO GENES ONLY
# ============================================================

comparison = f1_table.copy()


if "genes_only" in comparison.columns:


    baseline = (

        comparison[
            "genes_only"
        ]
    )


    original_columns = list(

        comparison.columns
    )


    for column in original_columns:


        if column == "genes_only":

            continue


        comparison[

            f"{column}_improvement"

        ] = (

            comparison[
                column
            ]

            - baseline
        )


comparison.to_csv(

    "per_cell_type_f1_improvement.csv"
)


print(
    "\n\n=========================================="
)


print(
    "F1 CHANGE RELATIVE TO GENES ONLY"
)


print(
    "==========================================\n"
)


improvement_columns = [

    column

    for column in comparison.columns

    if column.endswith(
        "_improvement"
    )
]


print(

    comparison[
        improvement_columns
    ]

    .round(3)
)


# ============================================================
# F1 COMPARISON BAR PLOT
# ============================================================

models = list(

    f1_table.columns
)


x = np.arange(

    len(
        f1_table.index
    )
)


width = (

    0.8

    / len(models)
)


fig, ax = plt.subplots(

    figsize=(
        18,
        8
    )
)


for i, model_name in enumerate(
    models
):


    offset = (

        i

        - (
            len(models) - 1
        ) / 2

    ) * width


    ax.bar(

        x + offset,

        f1_table[
            model_name
        ],

        width=width,

        label=model_name
    )


ax.set_xticks(
    x
)


ax.set_xticklabels(

    f1_table.index,

    rotation=60,

    ha="right"
)


ax.set_ylabel(
    "Mean F1 score"
)


ax.set_xlabel(
    "Cell type"
)


ax.set_ylim(
    0,
    1
)


ax.set_title(

    "Cell-type classification performance "
    "by feature set"
)


ax.legend(

    bbox_to_anchor=(
        1.02,
        1
    ),

    loc="upper left"
)


plt.tight_layout()


plt.savefig(

    "per_cell_type_f1_comparison.png",

    dpi=300
)


plt.close()


# ============================================================
# CONFUSION MATRICES
# ============================================================

for model_name, cm in confusion_totals.items():


    # --------------------------------------------------------
    # SAVE RAW COUNTS
    # --------------------------------------------------------

    cm_df = pd.DataFrame(

        cm,

        index=cell_types,

        columns=cell_types
    )


    cm_df.to_csv(

        f"confusion_matrix_"
        f"{model_name}_counts.csv"
    )


    # ========================================================
    # NORMALIZE EACH TRUE CELL TYPE
    # ========================================================

    # Every ROW represents one true cell type.
    #
    # Every row will sum to 1.
    #
    # This is useful because rare cell types otherwise become
    # nearly invisible compared with common cell types.

    row_sums = cm.sum(

        axis=1,

        keepdims=True
    )


    cm_normalized = np.divide(

        cm,

        row_sums,

        out=np.zeros_like(

            cm,

            dtype=float
        ),

        where=(

            row_sums != 0
        )
    )


    normalized_df = pd.DataFrame(

        cm_normalized,

        index=cell_types,

        columns=cell_types
    )


    normalized_df.to_csv(

        f"confusion_matrix_"
        f"{model_name}_normalized.csv"
    )


    # ========================================================
    # PLOT NORMALIZED CONFUSION MATRIX
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(
            14,
            12
        )
    )


    image = ax.imshow(

        cm_normalized,

        aspect="auto",

        vmin=0,

        vmax=1
    )


    ax.set_xticks(

        np.arange(
            len(cell_types)
        )
    )


    ax.set_yticks(

        np.arange(
            len(cell_types)
        )
    )


    ax.set_xticklabels(

        cell_types,

        rotation=90,

        fontsize=9
    )


    ax.set_yticklabels(

        cell_types,

        fontsize=9
    )


    ax.set_xlabel(

        "Predicted cell type"
    )


    ax.set_ylabel(

        "True cell type"
    )


    ax.set_title(

        f"Normalized confusion matrix\n"
        f"{model_name}"
    )


    fig.colorbar(

        image,

        ax=ax,

        label=(
            "Fraction of true cell type"
        )
    )


    plt.tight_layout()


    plt.savefig(

        f"confusion_matrix_"
        f"{model_name}.png",

        dpi=300
    )


    plt.close()


# ============================================================
# COORDINATE EFFECT BY CELL TYPE
# ============================================================

if (

    "genes_only"
    in f1_table.columns

    and

    "genes_volume_coordinates"
    in f1_table.columns
):


    coordinate_effect = pd.DataFrame(

        {

            "cell_type":
                f1_table.index,


            "genes_only_f1":

                f1_table[
                    "genes_only"
                ].values,


            "coordinates_f1":

                f1_table[
                    "genes_volume_coordinates"
                ].values,
        }
    )


    coordinate_effect[
        "f1_change"
    ] = (

        coordinate_effect[
            "coordinates_f1"
        ]

        -

        coordinate_effect[
            "genes_only_f1"
        ]
    )


    coordinate_effect = (

        coordinate_effect

        .sort_values(

            "f1_change",

            ascending=False
        )
    )


    coordinate_effect.to_csv(

        "coordinate_effect_by_cell_type.csv",

        index=False
    )


    print(
        "\n\n=========================================="
    )


    print(
        "CELL TYPES HELPED MOST BY COORDINATES"
    )


    print(
        "==========================================\n"
    )


    print(

        coordinate_effect

        .head(10)

        .round(3)
    )


    print(
        "\n\n=========================================="
    )


    print(
        "CELL TYPES HURT MOST BY COORDINATES"
    )


    print(
        "==========================================\n"
    )


    print(

        coordinate_effect

        .tail(10)

        .round(3)
    )


    # ========================================================
    # COORDINATE EFFECT PLOT
    # ========================================================

    plot_data = (

        coordinate_effect

        .sort_values(
            "f1_change"
        )
    )


    fig, ax = plt.subplots(

        figsize=(
            10,
            8
        )
    )


    ax.barh(

        plot_data[
            "cell_type"
        ],

        plot_data[
            "f1_change"
        ]
    )


    ax.axvline(

        0,

        linewidth=1
    )


    ax.set_xlabel(

        "Change in F1 after adding "
        "volume + coordinates"
    )


    ax.set_ylabel(
        "Cell type"
    )


    ax.set_title(

        "Which cell types benefit "
        "from spatial coordinates?"
    )


    plt.tight_layout()


    plt.savefig(

        "coordinate_f1_effect_by_cell_type.png",

        dpi=300
    )


    plt.close()


# ============================================================
# NEIGHBOR EFFECT BY CELL TYPE
# ============================================================

if (

    "genes_only"
    in f1_table.columns

    and

    "genes_volume_neighbors"
    in f1_table.columns
):


    neighbor_effect = pd.DataFrame(

        {

            "cell_type":
                f1_table.index,


            "genes_only_f1":

                f1_table[
                    "genes_only"
                ].values,


            "neighbors_f1":

                f1_table[
                    "genes_volume_neighbors"
                ].values,
        }
    )


    neighbor_effect[
        "f1_change"
    ] = (

        neighbor_effect[
            "neighbors_f1"
        ]

        -

        neighbor_effect[
            "genes_only_f1"
        ]
    )


    neighbor_effect = (

        neighbor_effect

        .sort_values(

            "f1_change",

            ascending=False
        )
    )


    neighbor_effect.to_csv(

        "neighbor_effect_by_cell_type.csv",

        index=False
    )


    print(
        "\n\n=========================================="
    )


    print(
        "CELL TYPES HELPED MOST BY NEIGHBORS"
    )


    print(
        "==========================================\n"
    )


    print(

        neighbor_effect

        .head(10)

        .round(3)
    )


    print(
        "\n\n=========================================="
    )


    print(
        "CELL TYPES HURT MOST BY NEIGHBORS"
    )


    print(
        "==========================================\n"
    )


    print(

        neighbor_effect

        .tail(10)

        .round(3)
    )


    # ========================================================
    # NEIGHBOR EFFECT PLOT
    # ========================================================

    plot_data = (

        neighbor_effect

        .sort_values(
            "f1_change"
        )
    )


    fig, ax = plt.subplots(

        figsize=(
            10,
            8
        )
    )


    ax.barh(

        plot_data[
            "cell_type"
        ],

        plot_data[
            "f1_change"
        ]
    )


    ax.axvline(

        0,

        linewidth=1
    )


    ax.set_xlabel(

        "Change in F1 after adding "
        "spatial neighbors"
    )


    ax.set_ylabel(
        "Cell type"
    )


    ax.set_title(

        "Which cell types benefit "
        "from spatial neighborhoods?"
    )


    plt.tight_layout()


    plt.savefig(

        "neighbor_f1_effect_by_cell_type.png",

        dpi=300
    )


    plt.close()


# ============================================================
# FINISHED
# ============================================================

print(
    "\n\n=========================================="
)


print(
    "ANALYSIS COMPLETE"
)


print(
    "=========================================="
)


print(
    "\nMain files created:"
)


print(
    "\nOverall model performance:"
)

print(
    "  classification_model_comparison_summary.csv"
)


print(
    "\nPer-cell-type performance:"
)

print(
    "  per_cell_type_performance_summary.csv"
)

print(
    "  per_cell_type_f1_comparison.csv"
)

print(
    "  per_cell_type_f1_improvement.csv"
)


print(
    "\nPlots:"
)

print(
    "  per_cell_type_f1_comparison.png"
)

print(
    "  coordinate_f1_effect_by_cell_type.png"
)

print(
    "  neighbor_f1_effect_by_cell_type.png"
)


print(
    "\nCoordinate analysis:"
)

print(
    "  coordinate_effect_by_cell_type.csv"
)


print(
    "\nNeighbor analysis:"
)

print(
    "  neighbor_effect_by_cell_type.csv"
)


print(
    "\nConfusion matrices:"
)

for model_name in confusion_totals:

    print(
        f"  confusion_matrix_"
        f"{model_name}.png"
    )