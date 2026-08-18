from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# SETTINGS
# ============================================================

# CHANGE THESE PATHS IF NEEDED

TRAIN_FILE = Path(
    "/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/data/full_train.csv"
)

TEST_FILE = Path(
    "/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/data/full_test.csv"
)

OUTPUT_FILE = Path(
    "/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/"
    "MERFISH_predictions.csv"
)

IMPORTANCE_FILE = Path(
    "/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/"
    "feature_importance.csv"
)


TARGET = "MERFISH_cell_type_annotation"

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"\nTraining rows: {len(train):,}")
print(f"Test rows:     {len(test):,}")

print(f"\nTraining columns: {len(train.columns)}")
print(f"Test columns:     {len(test.columns)}")


# ============================================================
# CHECK TARGET
# ============================================================

if TARGET not in train.columns:

    raise ValueError(
        f"{TARGET} was not found in the training data."
    )


print(
    f"\nNumber of cell types: "
    f"{train[TARGET].nunique()}"
)


# ============================================================
# FIND CELL ID COLUMN
# ============================================================

# The first column in your data appears to be a unique cell ID.
#
# Depending on how the CSV was created, pandas may call it:
#
# Unnamed: 0
#
# Or it may have some other name.
#
# We check whether the first column contains unique values.


first_column = train.columns[0]

ID_COLUMN = None


if str(first_column).startswith("Unnamed:"):

    ID_COLUMN = first_column

elif train[first_column].nunique() == len(train):

    ID_COLUMN = first_column


if ID_COLUMN is not None:

    print(
        f"\nDetected cell ID column: {ID_COLUMN}"
    )

    print(
        "This column will NOT be used to train the model."
    )

else:

    print(
        "\nNo obvious cell ID column detected."
    )


# ============================================================
# CREATE ENGINEERED FEATURES
# ============================================================

def add_engineered_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Region + Segment
    # --------------------------------------------------------

    if (
        "Region" in df.columns
        and "Segment" in df.columns
    ):

        df["region_segment"] = (
            df["Region"]
            .fillna("NA")
            .astype(str)
            +
            "__"
            +
            df["Segment"]
            .fillna("NA")
            .astype(str)
        )

    # --------------------------------------------------------
    # Region + Segment + Excitatory/Inhibitory
    # --------------------------------------------------------

    if (
        "Region" in df.columns
        and "Segment" in df.columns
        and "Excitatory_vs_Inhibitory" in df.columns
    ):

        df["region_segment_EI"] = (
            df["Region"]
            .fillna("NA")
            .astype(str)
            +
            "__"
            +
            df["Segment"]
            .fillna("NA")
            .astype(str)
            +
            "__"
            +
            df["Excitatory_vs_Inhibitory"]
            .fillna("NA")
            .astype(str)
        )

    return df


train = add_engineered_features(train)
test = add_engineered_features(test)


# ============================================================
# DEFINE METADATA COLUMNS
# ============================================================

# These columns are not gene-expression columns.

metadata_columns = [
    "Datasets",
    "volume",
    "center_x",
    "center_y",
    "Region",
    "Excitatory_vs_Inhibitory",
    "Segment",
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID",
    "region_segment",
    "region_segment_EI",
]


metadata_columns = [
    col
    for col in metadata_columns
    if col in train.columns
]


# ============================================================
# IDENTIFY GENE COLUMNS
# ============================================================

gene_columns = []


for col in train.columns:

    if col == TARGET:
        continue

    if col == ID_COLUMN:
        continue

    if col in metadata_columns:
        continue

    # Anything else is assumed to be one of the gene columns
    gene_columns.append(col)


print(
    f"\nNumber of gene features: "
    f"{len(gene_columns)}"
)


# ============================================================
# FEATURES FOR MAIN MODEL
# ============================================================

# These are the biological/spatial metadata features that
# seemed especially useful from your actual training rows.

biological_metadata = [
    "Region",
    "Segment",
    "Excitatory_vs_Inhibitory",
    "AP_position",
    "volume",
    "region_segment",
    "region_segment_EI",
]


biological_metadata = [
    col
    for col in biological_metadata
    if col in train.columns
    and col in test.columns
]


# Combine metadata + genes

features = biological_metadata + gene_columns


# Remove duplicates if there are any

features = list(
    dict.fromkeys(features)
)


# Only use features present in BOTH train and test

features = [
    col
    for col in features
    if col in train.columns
    and col in test.columns
]


print(
    f"\nTotal features used by model: "
    f"{len(features)}"
)


print("\nBiological metadata used:")

for col in biological_metadata:
    print(f"  {col}")


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

# Segment and Region are being treated as categories,
# NOT continuous numbers.

categorical_candidates = [
    "Region",
    "Segment",
    "Excitatory_vs_Inhibitory",
    "region_segment",
    "region_segment_EI",
]


# ============================================================
# PREPARE DATA FOR CATBOOST
# ============================================================

def prepare_catboost_data(df, feature_columns):

    output = df[feature_columns].copy()

    categorical_features = []


    for col in feature_columns:

        if col in categorical_candidates:

            # -----------------------------------------------
            # Categorical variables
            # -----------------------------------------------

            output[col] = (
                output[col]
                .fillna("MISSING")
                .astype(str)
            )

            categorical_features.append(col)

        else:

            # -----------------------------------------------
            # Numeric variables
            # -----------------------------------------------

            output[col] = pd.to_numeric(
                output[col],
                errors="coerce",
            )


    return output, categorical_features


# ============================================================
# CREATE TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating train/validation split...")


train_part, validation_part = train_test_split(
    train,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=train[TARGET],
)


print(
    f"Training subset:   "
    f"{len(train_part):,}"
)

print(
    f"Validation subset: "
    f"{len(validation_part):,}"
)


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

X_train, categorical_features = prepare_catboost_data(
    train_part,
    features,
)

X_validation, _ = prepare_catboost_data(
    validation_part,
    features,
)


y_train = train_part[TARGET]

y_validation = validation_part[TARGET]


print("\nCategorical features:")

for col in categorical_features:
    print(f"  {col}")


# ============================================================
# TRAIN FAST CATBOOST MODEL
# ============================================================

print("\n" + "=" * 70)

print("TRAINING CATBOOST")

print("=" * 70)


model = CatBoostClassifier(

    # Multiclass because there are many possible cell types
    loss_function="MultiClass",

    # Maximum number of trees.
    # Early stopping may stop much sooner.
    iterations=800,

    # Smaller depth = faster model
    depth=7,

    # Slightly faster learning than before
    learning_rate=0.10,

    random_seed=RANDOM_STATE,

    # Print progress every 50 iterations
    verbose=50,

    # Use every available CPU core
    thread_count=-1,

    # Don't create extra CatBoost files/folders
    allow_writing_files=False,

    # Stop if validation performance does not improve
    # for 50 iterations
    early_stopping_rounds=50,
)


model.fit(

    X_train,
    y_train,

    cat_features=categorical_features,

    # CatBoost watches this dataset for early stopping
    eval_set=(
        X_validation,
        y_validation,
    ),

    # Keep the best iteration rather than the final one
    use_best_model=True,
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_predictions = (
    model
    .predict(X_validation)
    .ravel()
)


# ============================================================
# EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(
    y_validation,
    validation_predictions,
)

balanced_accuracy = balanced_accuracy_score(
    y_validation,
    validation_predictions,
)

macro_f1 = f1_score(
    y_validation,
    validation_predictions,
    average="macro",
)


print("\n" + "=" * 70)

print("VALIDATION RESULTS")

print("=" * 70)


print(
    f"\nAccuracy:          "
    f"{accuracy:.4f}"
)

print(
    f"Balanced accuracy: "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Macro F1:          "
    f"{macro_f1:.4f}"
)


print(
    f"\nBest CatBoost iteration: "
    f"{model.get_best_iteration()}"
)


# ============================================================
# OPTIONAL DETAILED CLASS RESULTS
# ============================================================

print("\nClassification report:\n")

print(
    classification_report(
        y_validation,
        validation_predictions,
        zero_division=0,
    )
)


# ============================================================
# FEATURE IMPORTANCE FROM VALIDATION MODEL
# ============================================================

importance = pd.DataFrame(
    {
        "feature": features,
        "importance": model.get_feature_importance(),
    }
)


importance = importance.sort_values(
    "importance",
    ascending=False,
)


print("\n" + "=" * 70)

print("TOP 30 FEATURES")

print("=" * 70)


print(
    importance
    .head(30)
    .to_string(index=False)
)


# ============================================================
# DETERMINE BEST NUMBER OF ITERATIONS
# ============================================================

best_iteration = model.get_best_iteration()


# get_best_iteration() is zero-indexed

if best_iteration >= 0:

    final_iterations = best_iteration + 1

else:

    final_iterations = 400


print(
    f"\nFinal model will use "
    f"{final_iterations} iterations."
)


# ============================================================
# PREPARE FULL TRAINING DATA
# ============================================================

X_full, categorical_features = prepare_catboost_data(
    train,
    features,
)

X_test, _ = prepare_catboost_data(
    test,
    features,
)

y_full = train[TARGET]


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)

print("TRAINING FINAL MODEL ON ALL TRAINING DATA")

print("=" * 70)


final_model = CatBoostClassifier(

    loss_function="MultiClass",

    # Use the number of iterations selected by validation
    iterations=final_iterations,

    depth=7,

    learning_rate=0.10,

    random_seed=RANDOM_STATE,

    verbose=50,

    thread_count=-1,

    allow_writing_files=False,
)


final_model.fit(
    X_full,
    y_full,
    cat_features=categorical_features,
)


# ============================================================
# PREDICT TEST CELL TYPES
# ============================================================

print("\nPredicting test cell types...")


test_predictions = (
    final_model
    .predict(X_test)
    .ravel()
)


# ============================================================
# CREATE SUBMISSION FILE
# ============================================================

# Preserve the cell ID if one exists.

if (
    ID_COLUMN is not None
    and ID_COLUMN in test.columns
):

    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN],
            TARGET: test_predictions,
        }
    )

else:

    submission = pd.DataFrame(
        {
            TARGET: test_predictions
        }
    )


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)

print("PREDICTIONS SAVED")

print("=" * 70)


print(
    f"\n{OUTPUT_FILE.resolve()}"
)


# ============================================================
# SAVE FINAL FEATURE IMPORTANCE
# ============================================================

final_importance = pd.DataFrame(
    {
        "feature": features,
        "importance": final_model.get_feature_importance(),
    }
)


final_importance = final_importance.sort_values(
    "importance",
    ascending=False,
)


final_importance.to_csv(
    IMPORTANCE_FILE,
    index=False,
)


print(
    f"\nFeature importance saved to:\n"
    f"{IMPORTANCE_FILE.resolve()}"
)


# ============================================================
# SHOW PREDICTION COUNTS
# ============================================================

print("\n" + "=" * 70)

print("PREDICTED CELL TYPE COUNTS")

print("=" * 70)


print(
    submission[TARGET]
    .value_counts()
    .to_string()
)


# ============================================================
# SHOW FINAL TOP FEATURES
# ============================================================

print("\n" + "=" * 70)

print("FINAL TOP 30 FEATURES")

print("=" * 70)


print(
    final_importance
    .head(30)
    .to_string(index=False)
)


print("\nDone!")