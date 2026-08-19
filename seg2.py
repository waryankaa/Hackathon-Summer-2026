from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
)

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
)

from sklearn.feature_selection import (
    f_classif,
    VarianceThreshold,
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path(
    "/Users/abigailwaryanka/Desktop/"
    "Hackathon-Summer-2026/segments/segment_19.csv"
)

OUTPUT_DIR = Path(
    "/Users/abigailwaryanka/Desktop/"
    "Hackathon-Summer-2026/segment_19_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "MERFISH_cell_type_annotation"

RANDOM_STATE = 42

N_SPLITS = 5

N_REPEATS = 5


# ============================================================
# SAFE SELECT K BEST
# ============================================================

class SafeSelectKBest(
    BaseEstimator,
    TransformerMixin,
):

    def __init__(self, k=10):

        self.k = k


    def fit(self, X, y):

        X = np.asarray(X)

        number_of_features = X.shape[1]


        # Never request more genes than exist
        self.k_ = min(
            self.k,
            number_of_features,
        )


        # Calculate ANOVA F-score
        scores, p_values = f_classif(
            X,
            y,
        )


        # Replace invalid scores so they cannot be selected
        scores = np.nan_to_num(
            scores,
            nan=-np.inf,
            posinf=np.finfo(float).max,
            neginf=-np.inf,
        )


        self.scores_ = scores

        self.pvalues_ = p_values


        # Rank highest F-score first
        ranked_indices = np.argsort(
            scores
        )[::-1]


        self.selected_indices_ = np.sort(
            ranked_indices[
                :self.k_
            ]
        )


        return self


    def transform(self, X):

        X = np.asarray(X)

        return X[
            :,
            self.selected_indices_
        ]


    def get_support(
        self,
        indices=False,
    ):

        if indices:

            return self.selected_indices_


        mask = np.zeros(
            len(self.scores_),
            dtype=bool,
        )

        mask[
            self.selected_indices_
        ] = True


        return mask


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING SEGMENT DATA")
print("=" * 70)


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nNumber of cells: "
    f"{len(df):,}"
)

print(
    f"Number of columns: "
    f"{len(df.columns):,}"
)


# ============================================================
# CHECK TARGET
# ============================================================

if TARGET not in df.columns:

    raise ValueError(
        f"{TARGET} was not found."
    )


# ============================================================
# CELL TYPE COUNTS
# ============================================================

print("\nCell types:\n")

print(
    df[TARGET]
    .value_counts()
    .to_string()
)


cell_types = (
    df[TARGET]
    .dropna()
    .unique()
)


if len(cell_types) != 2:

    raise ValueError(
        f"This script expects exactly 2 cell types, "
        f"but found {len(cell_types)}."
    )


print(
    f"\nComparing:\n"
    f"  {cell_types[0]}\n"
    f"  {cell_types[1]}"
)


# ============================================================
# DEFINE METADATA COLUMNS
# ============================================================

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
    TARGET,
]


# ============================================================
# FIND CELL ID COLUMN
# ============================================================

first_column = df.columns[0]


if (
    str(first_column).startswith("Unnamed:")
    or df[first_column].nunique() == len(df)
):

    if first_column not in metadata_columns:

        metadata_columns.append(
            first_column
        )


    print(
        f"\nIgnoring cell ID column: "
        f"{first_column}"
    )


# ============================================================
# FIND GENE COLUMNS
# ============================================================

gene_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


print(
    f"\nPotential gene columns: "
    f"{len(gene_columns):,}"
)


# ============================================================
# CONVERT GENES TO NUMERIC
# ============================================================

for gene in gene_columns:

    df[gene] = pd.to_numeric(
        df[gene],
        errors="coerce",
    ).fillna(0)


# ============================================================
# REMOVE GENES CONSTANT ACROSS ENTIRE SEGMENT
# ============================================================

variable_genes = [
    gene
    for gene in gene_columns
    if df[gene].nunique() > 1
]


removed_globally = (
    len(gene_columns)
    - len(variable_genes)
)


print(
    f"Globally constant genes removed: "
    f"{removed_globally:,}"
)

print(
    f"Variable genes remaining: "
    f"{len(variable_genes):,}"
)


gene_columns = variable_genes


# ============================================================
# X AND Y
# ============================================================

X = df[
    gene_columns
].copy()


y = df[
    TARGET
].copy()


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(

    n_splits=N_SPLITS,

    n_repeats=N_REPEATS,

    random_state=RANDOM_STATE,
)


# ============================================================
# NUMBERS OF GENES TO TEST
# ============================================================

gene_counts = [
    1,
    2,
    3,
    5,
    10,
    20,
    40,
]


gene_counts = [
    k
    for k in gene_counts
    if k <= len(gene_columns)
]


print(
    "\nNumbers of genes being tested:"
)

print(
    gene_counts
)


# ============================================================
# CLASSIFIERS
# ============================================================

models = {

    "LogisticRegression":

        LogisticRegression(

            max_iter=5000,

            class_weight="balanced",

            random_state=RANDOM_STATE,
        ),


    "RandomForest":

        RandomForestClassifier(

            n_estimators=300,

            max_depth=4,

            class_weight="balanced",

            random_state=RANDOM_STATE,

            # cross_validate itself uses all cores
            n_jobs=1,
        ),


    "ExtraTrees":

        ExtraTreesClassifier(

            n_estimators=300,

            max_depth=4,

            class_weight="balanced",

            random_state=RANDOM_STATE,

            n_jobs=1,
        ),
}


# ============================================================
# FUNCTION TO BUILD A PIPELINE
# ============================================================

def build_pipeline(
    model_name,
    number_of_genes,
):

    classifier = clone(
        models[
            model_name
        ]
    )


    steps = [

        # ----------------------------------------------------
        # LOG TRANSFORM
        # ----------------------------------------------------

        (
            "log_transform",

            FunctionTransformer(
                np.log1p
            ),
        ),


        # ----------------------------------------------------
        # REMOVE CONSTANT GENES WITHIN CURRENT TRAINING FOLD
        # ----------------------------------------------------

        (
            "remove_constant",

            VarianceThreshold(
                threshold=0.0
            ),
        ),


        # ----------------------------------------------------
        # SELECT TOP GENES
        # ----------------------------------------------------

        (
            "feature_selection",

            SafeSelectKBest(
                k=number_of_genes
            ),
        ),
    ]


    # Logistic regression benefits from scaling

    if model_name == "LogisticRegression":

        steps.append(

            (
                "scaler",

                StandardScaler(),
            )
        )


    steps.append(

        (
            "classifier",

            classifier,
        )
    )


    return Pipeline(
        steps
    )


# ============================================================
# TEST EVERY MODEL
# ============================================================

results = []


print("\n" + "=" * 70)
print("TESTING INDIVIDUAL MODELS")
print("=" * 70)


for model_name in models:

    for k in gene_counts:

        print(
            f"\nTesting "
            f"{model_name} "
            f"with top {k} genes..."
        )


        pipeline = build_pipeline(
            model_name,
            k,
        )


        scores = cross_validate(

            pipeline,

            X,

            y,

            cv=cv,

            scoring={

                "accuracy":
                    "accuracy",

                "balanced_accuracy":
                    "balanced_accuracy",

                "f1_macro":
                    "f1_macro",
            },

            n_jobs=-1,

            return_train_score=False,
        )


        result = {

            "model":
                model_name,

            "number_of_genes":
                k,

            "accuracy_mean":
                scores[
                    "test_accuracy"
                ].mean(),

            "accuracy_sd":
                scores[
                    "test_accuracy"
                ].std(),

            "balanced_accuracy_mean":
                scores[
                    "test_balanced_accuracy"
                ].mean(),

            "balanced_accuracy_sd":
                scores[
                    "test_balanced_accuracy"
                ].std(),

            "macro_f1_mean":
                scores[
                    "test_f1_macro"
                ].mean(),

            "macro_f1_sd":
                scores[
                    "test_f1_macro"
                ].std(),
        }


        results.append(
            result
        )


        print(
            f"  Accuracy: "
            f"{result['accuracy_mean']:.3f}"
        )

        print(
            f"  Balanced accuracy: "
            f"{result['balanced_accuracy_mean']:.3f}"
        )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(

    [
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "number_of_genes",
    ],

    ascending=[
        False,
        False,
        True,
    ],
).reset_index(
    drop=True
)


print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(

    OUTPUT_DIR
    / "model_comparison.csv",

    index=False,
)


# ============================================================
# BEST INDIVIDUAL MODEL
# ============================================================

best = results_df.iloc[0]


best_model_name = (
    best["model"]
)


best_k = int(
    best["number_of_genes"]
)


print("\n" + "=" * 70)
print("BEST INDIVIDUAL METHOD")
print("=" * 70)


print(
    f"\nModel: "
    f"{best_model_name}"
)

print(
    f"Number of genes: "
    f"{best_k}"
)

print(
    f"Balanced accuracy: "
    f"{best['balanced_accuracy_mean']:.3f} "
    f"+/- "
    f"{best['balanced_accuracy_sd']:.3f}"
)

print(
    f"Accuracy: "
    f"{best['accuracy_mean']:.3f} "
    f"+/- "
    f"{best['accuracy_sd']:.3f}"
)


# ============================================================
# TOP THREE MODEL CONFIGURATIONS
# ============================================================

top_three = (
    results_df
    .head(3)
    .copy()
)


print("\n" + "=" * 70)
print("TOP THREE METHODS")
print("=" * 70)


print(
    top_three[
        [
            "model",
            "number_of_genes",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean",
        ]
    ].to_string(
        index=False
    )
)


top_three_methods = []


for _, row in top_three.iterrows():

    top_three_methods.append(

        {

            "model":
                row["model"],

            "number_of_genes":
                int(
                    row[
                        "number_of_genes"
                    ]
                ),
        }
    )


# ============================================================
# MAJORITY-VOTE ENSEMBLE
# ============================================================

print("\n" + "=" * 70)
print("TESTING 2-OUT-OF-3 ENSEMBLE")
print("=" * 70)


ensemble_fold_results = []

ensemble_prediction_records = []


# Make a NEW identical CV object so that we reproduce
# exactly the same style of cross-validation.

ensemble_cv = RepeatedStratifiedKFold(

    n_splits=N_SPLITS,

    n_repeats=N_REPEATS,

    random_state=RANDOM_STATE,
)


# ============================================================
# RUN EVERY FOLD
# ============================================================

for fold_number, (
    train_index,
    validation_index,
) in enumerate(
    ensemble_cv.split(
        X,
        y,
    ),
    start=1,
):


    X_train = X.iloc[
        train_index
    ]

    X_validation = X.iloc[
        validation_index
    ]


    y_train = y.iloc[
        train_index
    ]

    y_validation = y.iloc[
        validation_index
    ]


    # ========================================================
    # GET PREDICTIONS FROM ALL THREE METHODS
    # ========================================================

    model_predictions = []


    for method in top_three_methods:

        pipeline = build_pipeline(

            method[
                "model"
            ],

            method[
                "number_of_genes"
            ],
        )


        pipeline.fit(
            X_train,
            y_train,
        )


        predictions = pipeline.predict(
            X_validation
        )


        model_predictions.append(
            predictions
        )


    # ========================================================
    # MAJORITY VOTE FOR EACH CELL
    # ========================================================

    ensemble_predictions = []


    for i in range(
        len(y_validation)
    ):

        votes = [

            model_predictions[0][i],

            model_predictions[1][i],

            model_predictions[2][i],
        ]


        majority_vote = (

            Counter(
                votes
            )
            .most_common(1)[0][0]
        )


        ensemble_predictions.append(
            majority_vote
        )


    ensemble_predictions = np.array(
        ensemble_predictions
    )


    # ========================================================
    # FOLD METRICS
    # ========================================================

    fold_accuracy = accuracy_score(

        y_validation,

        ensemble_predictions,
    )


    fold_balanced_accuracy = (
        balanced_accuracy_score(

            y_validation,

            ensemble_predictions,
        )
    )


    fold_macro_f1 = f1_score(

        y_validation,

        ensemble_predictions,

        average="macro",
    )


    ensemble_fold_results.append(

        {

            "fold":
                fold_number,

            "accuracy":
                fold_accuracy,

            "balanced_accuracy":
                fold_balanced_accuracy,

            "macro_f1":
                fold_macro_f1,
        }
    )


    # ========================================================
    # SAVE PER-CELL PREDICTIONS
    # ========================================================

    for i, original_index in enumerate(
        validation_index
    ):

        ensemble_prediction_records.append(

            {

                "fold":
                    fold_number,

                "row_index":
                    original_index,

                "true_cell_type":
                    y_validation.iloc[i],

                "model_1":
                    model_predictions[0][i],

                "model_2":
                    model_predictions[1][i],

                "model_3":
                    model_predictions[2][i],

                "ensemble_prediction":
                    ensemble_predictions[i],

                "correct":
                    (
                        ensemble_predictions[i]
                        ==
                        y_validation.iloc[i]
                    ),
            }
        )


# ============================================================
# ENSEMBLE RESULTS DATAFRAME
# ============================================================

ensemble_results_df = pd.DataFrame(
    ensemble_fold_results
)


ensemble_predictions_df = pd.DataFrame(
    ensemble_prediction_records
)


# ============================================================
# MEAN ENSEMBLE PERFORMANCE
# ============================================================

ensemble_accuracy_mean = (
    ensemble_results_df[
        "accuracy"
    ].mean()
)


ensemble_accuracy_sd = (
    ensemble_results_df[
        "accuracy"
    ].std()
)


ensemble_balanced_mean = (
    ensemble_results_df[
        "balanced_accuracy"
    ].mean()
)


ensemble_balanced_sd = (
    ensemble_results_df[
        "balanced_accuracy"
    ].std()
)


ensemble_f1_mean = (
    ensemble_results_df[
        "macro_f1"
    ].mean()
)


ensemble_f1_sd = (
    ensemble_results_df[
        "macro_f1"
    ].std()
)


# ============================================================
# PRINT ENSEMBLE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ENSEMBLE RESULTS")
print("=" * 70)


print(
    f"\nAccuracy: "
    f"{ensemble_accuracy_mean:.3f} "
    f"+/- "
    f"{ensemble_accuracy_sd:.3f}"
)


print(
    f"Balanced accuracy: "
    f"{ensemble_balanced_mean:.3f} "
    f"+/- "
    f"{ensemble_balanced_sd:.3f}"
)


print(
    f"Macro F1: "
    f"{ensemble_f1_mean:.3f} "
    f"+/- "
    f"{ensemble_f1_sd:.3f}"
)


# ============================================================
# COMPARE ENSEMBLE VS BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("ENSEMBLE VS BEST INDIVIDUAL MODEL")
print("=" * 70)


best_accuracy = (
    best[
        "accuracy_mean"
    ]
)


best_balanced = (
    best[
        "balanced_accuracy_mean"
    ]
)


print(
    f"\nBest individual accuracy: "
    f"{best_accuracy:.3f}"
)


print(
    f"Ensemble accuracy:        "
    f"{ensemble_accuracy_mean:.3f}"
)


print(
    f"Difference:               "
    f"{ensemble_accuracy_mean - best_accuracy:+.3f}"
)


print(
    f"\nBest individual balanced accuracy: "
    f"{best_balanced:.3f}"
)


print(
    f"Ensemble balanced accuracy:        "
    f"{ensemble_balanced_mean:.3f}"
)


print(
    f"Difference:                        "
    f"{ensemble_balanced_mean - best_balanced:+.3f}"
)


# ============================================================
# SAVE ENSEMBLE RESULTS
# ============================================================

ensemble_summary = pd.DataFrame(

    {

        "metric": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],

        "mean": [
            ensemble_accuracy_mean,
            ensemble_balanced_mean,
            ensemble_f1_mean,
        ],

        "sd": [
            ensemble_accuracy_sd,
            ensemble_balanced_sd,
            ensemble_f1_sd,
        ],
    }
)


ensemble_summary.to_csv(

    OUTPUT_DIR
    / "ensemble_results.csv",

    index=False,
)


ensemble_predictions_df.to_csv(

    OUTPUT_DIR
    / "ensemble_cell_predictions.csv",

    index=False,
)


# ============================================================
# SHOW ENSEMBLE CONFUSION MATRIX
# ============================================================

all_true = (
    ensemble_predictions_df[
        "true_cell_type"
    ]
)


all_predicted = (
    ensemble_predictions_df[
        "ensemble_prediction"
    ]
)


cm = confusion_matrix(

    all_true,

    all_predicted,

    labels=cell_types,
)


confusion_df = pd.DataFrame(

    cm,

    index=[
        f"Actual_{x}"
        for x in cell_types
    ],

    columns=[
        f"Predicted_{x}"
        for x in cell_types
    ],
)


print("\n" + "=" * 70)
print("ENSEMBLE CONFUSION MATRIX")
print("=" * 70)


print(
    "\n",
    confusion_df,
)


confusion_df.to_csv(

    OUTPUT_DIR
    / "ensemble_confusion_matrix.csv"
)


# ============================================================
# HOW OFTEN DO ALL THREE MODELS AGREE?
# ============================================================

ensemble_predictions_df[
    "all_three_agree"
] = (

    (
        ensemble_predictions_df[
            "model_1"
        ]
        ==
        ensemble_predictions_df[
            "model_2"
        ]
    )

    &

    (
        ensemble_predictions_df[
            "model_2"
        ]
        ==
        ensemble_predictions_df[
            "model_3"
        ]
    )
)


agreement_rate = (

    ensemble_predictions_df[
        "all_three_agree"
    ].mean()
)


print(
    f"\nAll three models agree on "
    f"{agreement_rate:.1%} "
    f"of predictions."
)


# ============================================================
# ACCURACY WHEN ALL THREE AGREE
# ============================================================

agreement_rows = (

    ensemble_predictions_df[
        ensemble_predictions_df[
            "all_three_agree"
        ]
    ]
)


if len(
    agreement_rows
) > 0:

    agreement_accuracy = (

        agreement_rows[
            "correct"
        ].mean()
    )


    print(
        f"Accuracy when all 3 agree: "
        f"{agreement_accuracy:.1%}"
    )


# ============================================================
# ACCURACY WHEN ONLY 2 OF 3 AGREE
# ============================================================

split_vote_rows = (

    ensemble_predictions_df[
        ~ensemble_predictions_df[
            "all_three_agree"
        ]
    ]
)


if len(
    split_vote_rows
) > 0:

    split_vote_accuracy = (

        split_vote_rows[
            "correct"
        ].mean()
    )


    print(
        f"Accuracy when vote is 2-to-1: "
        f"{split_vote_accuracy:.1%}"
    )


# ============================================================
# GENE RANKING
# ============================================================

print("\n" + "=" * 70)
print("RANKING GENES")
print("=" * 70)


X_log = np.log1p(
    X
)


variance_filter = VarianceThreshold(
    threshold=0.0
)


X_variable = (
    variance_filter
    .fit_transform(
        X_log
    )
)


variance_mask = (
    variance_filter
    .get_support()
)


genes_after_variance = np.array(
    gene_columns
)[
    variance_mask
]


f_scores, p_values = f_classif(

    X_variable,

    y,
)


gene_ranking = pd.DataFrame(

    {

        "gene":
            genes_after_variance,

        "F_score":
            f_scores,

        "p_value":
            p_values,
    }
)


gene_ranking = gene_ranking.sort_values(

    "F_score",

    ascending=False,
)


# ============================================================
# ADD MEAN EXPRESSION FOR BOTH CELL TYPES
# ============================================================

type_1 = cell_types[0]

type_2 = cell_types[1]


type_1_means = (

    df[
        df[TARGET] == type_1
    ][gene_columns]
    .mean()
)


type_2_means = (

    df[
        df[TARGET] == type_2
    ][gene_columns]
    .mean()
)


gene_ranking[
    f"{type_1}_mean"
] = (

    gene_ranking[
        "gene"
    ]
    .map(
        type_1_means
    )
)


gene_ranking[
    f"{type_2}_mean"
] = (

    gene_ranking[
        "gene"
    ]
    .map(
        type_2_means
    )
)


gene_ranking[
    "mean_difference"
] = (

    gene_ranking[
        f"{type_1}_mean"
    ]

    -

    gene_ranking[
        f"{type_2}_mean"
    ]
)


# ============================================================
# SHOW TOP 30 GENES
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 GENES")
print("=" * 70)


print(
    gene_ranking
    .head(30)
    .to_string(
        index=False
    )
)


gene_ranking.to_csv(

    OUTPUT_DIR
    / "gene_ranking.csv",

    index=False,
)


# ============================================================
# FIT BEST INDIVIDUAL MODEL ON ALL DATA
# ============================================================

best_pipeline = build_pipeline(

    best_model_name,

    best_k,
)


best_pipeline.fit(
    X,
    y,
)


# ============================================================
# FIND GENES USED BY BEST MODEL
# ============================================================

variance_filter = (

    best_pipeline
    .named_steps[
        "remove_constant"
    ]
)


variance_mask = (

    variance_filter
    .get_support()
)


genes_after_variance = np.array(
    gene_columns
)[
    variance_mask
]


selector = (

    best_pipeline
    .named_steps[
        "feature_selection"
    ]
)


selected_mask = (

    selector
    .get_support()
)


selected_genes = (

    genes_after_variance[
        selected_mask
    ]
)


print("\n" + "=" * 70)
print("GENES USED BY BEST INDIVIDUAL MODEL")
print("=" * 70)


for gene in selected_genes:

    print(
        gene
    )


pd.DataFrame(

    {
        "gene":
            selected_genes
    }

).to_csv(

    OUTPUT_DIR
    / "best_model_genes.csv",

    index=False,
)


# ============================================================
# SIMPLE ONE-GENE DECISION RULE
# ============================================================

simple_tree = DecisionTreeClassifier(

    max_depth=1,

    class_weight="balanced",

    random_state=RANDOM_STATE,
)


simple_tree.fit(
    X,
    y,
)


tree_rule = export_text(

    simple_tree,

    feature_names=list(
        X.columns
    ),
)


print("\n" + "=" * 70)
print("BEST SIMPLE ONE-GENE RULE")
print("=" * 70)


print(
    tree_rule
)


with open(

    OUTPUT_DIR
    / "simple_rule.txt",

    "w",

) as file:

    file.write(
        tree_rule
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)


print(
    f"\nResults saved in:\n"
    f"{OUTPUT_DIR}"
)


print(
    "\nFiles created:"
)

print(
    "  model_comparison.csv"
)

print(
    "  ensemble_results.csv"
)

print(
    "  ensemble_cell_predictions.csv"
)

print(
    "  ensemble_confusion_matrix.csv"
)

print(
    "  gene_ranking.csv"
)

print(
    "  best_model_genes.csv"
)

print(
    "  simple_rule.txt"
)