from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = Path("/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/segments/segment_NA.csv")
TEST_FILE = Path("/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/data/full_test.csv")

OUTPUT_DIR = Path("/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/segment_NA_frequency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "MERFISH_cell_type_annotation"
DATASET_COL = "Datasets"
AP_COL = "AP_position"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5

GENE_COUNTS = [1, 2, 3, 5, 10, 20, 40]

# alpha = 0 means completely ignore the frequency prior.
ALPHAS = [0.0, 0.25, 0.5, 1.0]

# Controls how strongly small Dataset + AP groups are shrunk
# toward the overall NA cell-type frequencies.
#
# Larger number = trust tiny groups less.
PRIOR_STRENGTH = 20


# ============================================================
# SAFE FEATURE SELECTION
# ============================================================

class SafeSelectKBest(BaseEstimator, TransformerMixin):

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)

        if X.shape[1] == 0:
            raise ValueError("No genes remain after variance filtering.")

        self.k_ = min(self.k, X.shape[1])

        scores, p_values = f_classif(X, y)

        self.scores_ = np.nan_to_num(
            scores,
            nan=-np.inf,
            posinf=np.finfo(float).max,
            neginf=-np.inf,
        )

        self.pvalues_ = p_values

        ranked = np.argsort(self.scores_)[::-1]
        self.selected_indices_ = np.sort(ranked[:self.k_])

        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]

    def get_support(self, indices=False):

        if indices:
            return self.selected_indices_

        mask = np.zeros(len(self.scores_), dtype=bool)
        mask[self.selected_indices_] = True

        return mask


# ============================================================
# HELPERS
# ============================================================

def clean_group_value(value):

    if pd.isna(value):
        return "__NA__"

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

    except (ValueError, TypeError):
        pass

    return str(value).strip()


def prepare_keys(df):

    df = df.copy()

    df["_dataset_key"] = df[DATASET_COL].apply(clean_group_value)
    df["_ap_key"] = df[AP_COL].apply(clean_group_value)

    return df


def log1p_nonnegative(X):
    X = np.asarray(X, dtype=float)
    return np.log1p(np.clip(X, 0, None))


def find_id_column(df):

    first_column = df.columns[0]

    if str(first_column).startswith("Unnamed:") or df[first_column].nunique() == len(df):
        return first_column

    possible_ids = [
        "identity_number",
        "Identity",
        "identity",
        "cell_id",
        "Cell_ID",
        "ID",
        "id",
    ]

    return next((column for column in possible_ids if column in df.columns), None)


# ============================================================
# MODELS
# ============================================================

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
}


def build_pipeline(model_name, number_of_genes):

    steps = [
        ("log_transform", FunctionTransformer(log1p_nonnegative, validate=False)),
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SafeSelectKBest(k=number_of_genes)),
    ]

    if model_name == "LogisticRegression":
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", clone(models[model_name])))

    return Pipeline(steps)


# ============================================================
# CREATE SMOOTHED DATASET + AP FREQUENCY PRIORS
# ============================================================

def make_frequency_priors(reference_df, classes):

    global_counts = reference_df[TARGET].value_counts()
    total_cells = len(reference_df)

    global_frequency = {
        cell_type: global_counts.get(cell_type, 0) / total_cells
        for cell_type in classes
    }

    group_priors = {}
    group_counts = {}

    for key, group in reference_df.groupby(["_dataset_key", "_ap_key"]):

        counts = group[TARGET].value_counts()
        n = len(group)

        group_counts[key] = {
            cell_type: int(counts.get(cell_type, 0))
            for cell_type in classes
        }

        group_priors[key] = {
            cell_type: (
                counts.get(cell_type, 0)
                + PRIOR_STRENGTH * global_frequency[cell_type]
            ) / (n + PRIOR_STRENGTH)
            for cell_type in classes
        }

    return global_frequency, group_priors, group_counts


# ============================================================
# FREQUENCY-WEIGHTED PREDICTION
# ============================================================

def frequency_weighted_predict(model, reference_df, prediction_df, gene_columns, alpha):

    probabilities = model.predict_proba(prediction_df[gene_columns])
    classes = model.named_steps["classifier"].classes_

    global_frequency, group_priors, _ = make_frequency_priors(
        reference_df,
        classes,
    )

    records = []

    for i, (_, row) in enumerate(prediction_df.iterrows()):

        key = (
            row["_dataset_key"],
            row["_ap_key"],
        )

        model_probabilities = probabilities[i]

        normal_index = np.argmax(model_probabilities)
        normal_prediction = classes[normal_index]
        normal_probability = model_probabilities[normal_index]

        if key in group_priors:
            priors = np.array([group_priors[key][cell_type] for cell_type in classes])
            prior_source = "dataset_AP"
        else:
            priors = np.array([global_frequency[cell_type] for cell_type in classes])
            prior_source = "global_frequency"

        weighted_scores = model_probabilities * np.power(priors, alpha)

        if weighted_scores.sum() > 0:
            weighted_probabilities = weighted_scores / weighted_scores.sum()
        else:
            weighted_probabilities = model_probabilities

        weighted_index = np.argmax(weighted_probabilities)
        weighted_prediction = classes[weighted_index]

        records.append({
            "normal_prediction": normal_prediction,
            "weighted_prediction": weighted_prediction,
            "prediction_changed": normal_prediction != weighted_prediction,
            "normal_probability": normal_probability,
            "model_probability_for_weighted_class": model_probabilities[weighted_index],
            "training_frequency_prior": priors[weighted_index],
            "weighted_probability": weighted_probabilities[weighted_index],
            "prior_source": prior_source,
        })

    return pd.DataFrame(records, index=prediction_df.index)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ============================================================
# TEST ONE MODEL + GENE COUNT ACROSS ALL ALPHAS
# ============================================================

def evaluate_model_configuration(df, gene_columns, model_name, k):

    class_counts = df[TARGET].value_counts()
    n_splits = min(N_SPLITS, int(class_counts.min()))

    if n_splits < 2:
        return []

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    alpha_results = {
        alpha: []
        for alpha in ALPHAS
    }

    for train_index, validation_index in cv.split(df, df[TARGET]):

        train_fold = df.iloc[train_index].copy()
        validation_fold = df.iloc[validation_index].copy()

        pipeline = build_pipeline(model_name, k)

        try:
            pipeline.fit(train_fold[gene_columns], train_fold[TARGET])

        except Exception:
            return []

        for alpha in ALPHAS:

            prediction_info = frequency_weighted_predict(
                pipeline,
                train_fold,
                validation_fold,
                gene_columns,
                alpha,
            )

            metrics = calculate_metrics(
                validation_fold[TARGET],
                prediction_info["weighted_prediction"],
            )

            alpha_results[alpha].append(metrics)

    results = []

    for alpha in ALPHAS:

        fold_df = pd.DataFrame(alpha_results[alpha])

        results.append({
            "model": model_name,
            "number_of_genes": k,
            "alpha": alpha,
            "accuracy_mean": fold_df["accuracy"].mean(),
            "accuracy_sd": fold_df["accuracy"].std(),
            "balanced_accuracy_mean": fold_df["balanced_accuracy"].mean(),
            "balanced_accuracy_sd": fold_df["balanced_accuracy"].std(),
            "macro_f1_mean": fold_df["macro_f1"].mean(),
            "macro_f1_sd": fold_df["macro_f1"].std(),
        })

    return results


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

if TARGET not in train_df.columns:
    raise ValueError(f"{TARGET} was not found in training data.")

for column in [DATASET_COL, AP_COL]:

    if column not in train_df.columns:
        raise ValueError(f"{column} was not found in training data.")

    if column not in test_df.columns:
        raise ValueError(f"{column} was not found in test data.")

train_df = train_df[train_df[TARGET].notna()].copy().reset_index(drop=True)


# ============================================================
# KEEP ONLY NA CELLS FROM TEST
# ============================================================

if "Segment" in test_df.columns:

    segment = test_df["Segment"]

    na_mask = (
        segment.isna()
        | segment.astype(str).str.strip().str.upper().eq("NA")
    )

    test_df = test_df[na_mask].copy().reset_index(drop=True)


# ============================================================
# DATASET + AP KEYS
# ============================================================

train_df = prepare_keys(train_df)
test_df = prepare_keys(test_df)

print(f"\nTraining NA cells: {len(train_df):,}")
print(f"Test NA cells: {len(test_df):,}")


# ============================================================
# ID COLUMNS
# ============================================================

TRAIN_ID = find_id_column(train_df)
TEST_ID = find_id_column(test_df)

if TRAIN_ID is None:
    train_df["_identity_number"] = train_df.index
    TRAIN_ID = "_identity_number"

if TEST_ID is None:
    test_df["_identity_number"] = test_df.index
    TEST_ID = "_identity_number"

print(f"Training identity column: {TRAIN_ID}")
print(f"Test identity column: {TEST_ID}")


# ============================================================
# CELL TYPES
# ============================================================

cell_types = list(pd.unique(train_df[TARGET]))
class_counts = train_df[TARGET].value_counts()

print("\nCell types:\n")
print(class_counts.to_string())

smallest_class = int(class_counts.min())
actual_splits = min(N_SPLITS, smallest_class)

if actual_splits < 2:
    raise ValueError("At least one cell type has fewer than 2 cells.")


# ============================================================
# FIND GENE COLUMNS
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
    "cell_type_frequency",
    "normalized_frequency",
    "frequency",
    TARGET,
    TRAIN_ID,
    "_dataset_key",
    "_ap_key",
]

gene_columns = [
    column
    for column in train_df.columns
    if column not in metadata_columns
    and column in test_df.columns
]

for gene in gene_columns:

    train_df[gene] = pd.to_numeric(
        train_df[gene],
        errors="coerce",
    ).fillna(0)

    test_df[gene] = pd.to_numeric(
        test_df[gene],
        errors="coerce",
    ).fillna(0)

variable_genes = [
    gene
    for gene in gene_columns
    if train_df[gene].nunique() > 1
]

print(f"\nPotential genes: {len(gene_columns):,}")
print(f"Constant genes removed: {len(gene_columns) - len(variable_genes):,}")
print(f"Variable genes remaining: {len(variable_genes):,}")

gene_columns = variable_genes

if len(gene_columns) == 0:
    raise ValueError("No variable genes remain.")

gene_counts = [
    k
    for k in GENE_COUNTS
    if k <= len(gene_columns)
]

print(f"Numbers of genes tested: {gene_counts}")
print(f"Frequency alpha values tested: {ALPHAS}")


# ============================================================
# SAVE TRAINING FREQUENCY PRIORS
# ============================================================

global_frequency, group_priors, group_counts = make_frequency_priors(
    train_df,
    cell_types,
)

frequency_records = []

for key, priors in group_priors.items():

    dataset, ap = key

    group_n = sum(group_counts[key].values())

    for cell_type in cell_types:

        frequency_records.append({
            "dataset": dataset,
            "AP_position": ap,
            "cell_type": cell_type,
            "cell_count": group_counts[key][cell_type],
            "group_total": group_n,
            "raw_group_frequency": group_counts[key][cell_type] / group_n,
            "global_frequency": global_frequency[cell_type],
            "smoothed_frequency_prior": priors[cell_type],
        })

frequency_df = pd.DataFrame(frequency_records)

frequency_df.to_csv(
    OUTPUT_DIR / "training_frequency_priors.csv",
    index=False,
)


# ============================================================
# TEST ALL MODELS + GENES + FREQUENCY WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("TESTING MODELS AND FREQUENCY WEIGHTS")
print("=" * 70)

results = []

for model_name in models:

    for k in gene_counts:

        print(f"\nTesting {model_name} with {k} genes...")

        configuration_results = evaluate_model_configuration(
            train_df,
            gene_columns,
            model_name,
            k,
        )

        results.extend(configuration_results)

        for result in configuration_results:

            print(
                f"  alpha={result['alpha']}: "
                f"balanced={result['balanced_accuracy_mean']:.3f}, "
                f"accuracy={result['accuracy_mean']:.3f}"
            )


# ============================================================
# MODEL RESULTS
# ============================================================

if len(results) == 0:
    raise ValueError("No model configuration could be evaluated.")

results_df = pd.DataFrame(results).sort_values(
    [
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "accuracy_mean",
        "number_of_genes",
        "alpha",
    ],
    ascending=[
        False,
        False,
        False,
        True,
        True,
    ],
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_DIR / "frequency_weighted_model_comparison.csv",
    index=False,
)

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# BEST CONFIGURATION
# ============================================================

best = results_df.iloc[0]

best_model_name = best["model"]
best_k = int(best["number_of_genes"])
best_alpha = float(best["alpha"])

print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(f"\nModel: {best_model_name}")
print(f"Number of genes: {best_k}")
print(f"Frequency alpha: {best_alpha}")
print(f"Accuracy: {best['accuracy_mean']:.3f} +/- {best['accuracy_sd']:.3f}")
print(f"Balanced accuracy: {best['balanced_accuracy_mean']:.3f} +/- {best['balanced_accuracy_sd']:.3f}")
print(f"Macro F1: {best['macro_f1_mean']:.3f} +/- {best['macro_f1_sd']:.3f}")

if best_alpha == 0:
    print("\nFrequency weighting did NOT improve the selected model.")
else:
    print(f"\nFrequency weighting improved the selected configuration enough to choose alpha={best_alpha}.")


# ============================================================
# COMPARE BEST WEIGHTED CONFIGURATION TO SAME MODEL WITH ALPHA=0
# ============================================================

same_model_baseline = results_df[
    (results_df["model"] == best_model_name)
    & (results_df["number_of_genes"] == best_k)
    & (results_df["alpha"] == 0)
]

if len(same_model_baseline) > 0:

    baseline = same_model_baseline.iloc[0]

    print("\nSame model without frequency weighting:")
    print(f"  Balanced accuracy: {baseline['balanced_accuracy_mean']:.3f}")
    print(f"  Weighted balanced accuracy: {best['balanced_accuracy_mean']:.3f}")
    print(f"  Difference: {best['balanced_accuracy_mean'] - baseline['balanced_accuracy_mean']:+.3f}")


# ============================================================
# ONE HELD-OUT PREDICTION PER TRAINING CELL
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED CELL TYPES")
print("=" * 70)

prediction_cv = StratifiedKFold(
    n_splits=actual_splits,
    shuffle=True,
    random_state=RANDOM_STATE,
)

prediction_records = []

for fold_number, (train_index, validation_index) in enumerate(
    prediction_cv.split(train_df, train_df[TARGET]),
    start=1,
):

    train_fold = train_df.iloc[train_index].copy()
    validation_fold = train_df.iloc[validation_index].copy()

    pipeline = build_pipeline(
        best_model_name,
        best_k,
    )

    pipeline.fit(
        train_fold[gene_columns],
        train_fold[TARGET],
    )

    prediction_info = frequency_weighted_predict(
        pipeline,
        train_fold,
        validation_fold,
        gene_columns,
        best_alpha,
    )

    for i, original_index in enumerate(validation_index):

        info = prediction_info.iloc[i]
        actual = validation_fold.iloc[i][TARGET]

        prediction_records.append({
            "identity_number": train_df.loc[original_index, TRAIN_ID],
            "Datasets": validation_fold.iloc[i][DATASET_COL],
            "AP_position": validation_fold.iloc[i][AP_COL],
            "normal_prediction": info["normal_prediction"],
            "predicted_cell_type": info["weighted_prediction"],
            "actual_cell_type": actual,
            "prediction_changed": info["prediction_changed"],
            "normal_correct": info["normal_prediction"] == actual,
            "weighted_correct": info["weighted_prediction"] == actual,
            "normal_probability": info["normal_probability"],
            "model_probability_for_weighted_class": info["model_probability_for_weighted_class"],
            "training_frequency_prior": info["training_frequency_prior"],
            "weighted_probability": info["weighted_probability"],
            "prior_source": info["prior_source"],
            "fold": fold_number,
        })

prediction_df = pd.DataFrame(prediction_records).sort_values(
    "identity_number"
).reset_index(drop=True)

prediction_df.to_csv(
    OUTPUT_DIR / "best_model_identity_predictions.csv",
    index=False,
)

prediction_df[
    [
        "identity_number",
        "predicted_cell_type",
        "actual_cell_type",
    ]
].to_csv(
    OUTPUT_DIR / "best_model_identity_predictions_simple.csv",
    index=False,
)


# ============================================================
# ACTUAL VS PREDICTED PERFORMANCE
# ============================================================

normal_accuracy = prediction_df["normal_correct"].mean()
weighted_accuracy = prediction_df["weighted_correct"].mean()

normal_balanced = balanced_accuracy_score(
    prediction_df["actual_cell_type"],
    prediction_df["normal_prediction"],
)

weighted_balanced = balanced_accuracy_score(
    prediction_df["actual_cell_type"],
    prediction_df["predicted_cell_type"],
)

normal_f1 = f1_score(
    prediction_df["actual_cell_type"],
    prediction_df["normal_prediction"],
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    prediction_df["actual_cell_type"],
    prediction_df["predicted_cell_type"],
    average="macro",
    zero_division=0,
)

print(f"\nNormal accuracy: {normal_accuracy:.3%}")
print(f"Weighted accuracy: {weighted_accuracy:.3%}")
print(f"Difference: {weighted_accuracy - normal_accuracy:+.3%}")

print(f"\nNormal balanced accuracy: {normal_balanced:.3%}")
print(f"Weighted balanced accuracy: {weighted_balanced:.3%}")
print(f"Difference: {weighted_balanced - normal_balanced:+.3%}")

print(f"\nNormal macro F1: {normal_f1:.3%}")
print(f"Weighted macro F1: {weighted_f1:.3%}")

changed_df = prediction_df[
    prediction_df["prediction_changed"]
].copy()

print(f"\nPredictions changed by frequency weighting: {len(changed_df):,} / {len(prediction_df):,}")

if len(changed_df) > 0:
    print(f"Normal accuracy among changed cells: {changed_df['normal_correct'].mean():.3%}")
    print(f"Weighted accuracy among changed cells: {changed_df['weighted_correct'].mean():.3%}")

changed_df.to_csv(
    OUTPUT_DIR / "predictions_changed_by_frequency.csv",
    index=False,
)


# ============================================================
# INCORRECT WEIGHTED PREDICTIONS
# ============================================================

incorrect_df = prediction_df[
    ~prediction_df["weighted_correct"]
].copy()

incorrect_df.to_csv(
    OUTPUT_DIR / "incorrect_weighted_predictions.csv",
    index=False,
)


# ============================================================
# ACCURACY BY CELL TYPE
# ============================================================

accuracy_by_type = (
    prediction_df
    .groupby("actual_cell_type")
    .agg(
        number_of_cells=("weighted_correct", "size"),
        normal_correct=("normal_correct", "sum"),
        weighted_correct=("weighted_correct", "sum"),
        normal_accuracy=("normal_correct", "mean"),
        weighted_accuracy=("weighted_correct", "mean"),
    )
    .reset_index()
)

accuracy_by_type["difference"] = (
    accuracy_by_type["weighted_accuracy"]
    - accuracy_by_type["normal_accuracy"]
)

accuracy_by_type = accuracy_by_type.sort_values(
    "weighted_accuracy"
)

accuracy_by_type.to_csv(
    OUTPUT_DIR / "accuracy_by_cell_type.csv",
    index=False,
)

print("\n" + "=" * 70)
print("ACCURACY BY CELL TYPE")
print("=" * 70)

print(accuracy_by_type.to_string(index=False))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    prediction_df["actual_cell_type"],
    prediction_df["predicted_cell_type"],
    labels=cell_types,
)

confusion_df = pd.DataFrame(
    cm,
    index=[f"Actual_{cell_type}" for cell_type in cell_types],
    columns=[f"Predicted_{cell_type}" for cell_type in cell_types],
)

confusion_df.to_csv(
    OUTPUT_DIR / "weighted_confusion_matrix.csv"
)


# ============================================================
# FIT FINAL MODEL ON ALL NA TRAINING CELLS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

final_model = build_pipeline(
    best_model_name,
    best_k,
)

final_model.fit(
    train_df[gene_columns],
    train_df[TARGET],
)


# ============================================================
# GENES USED BY FINAL MODEL
# ============================================================

variance_mask = final_model.named_steps[
    "remove_constant"
].get_support()

genes_after_variance = np.array(
    gene_columns
)[variance_mask]

selector = final_model.named_steps[
    "feature_selection"
]

selected_genes = genes_after_variance[
    selector.get_support()
]

print("\nSelected genes:")

for gene in selected_genes:
    print(f"  {gene}")

pd.DataFrame({
    "gene": selected_genes
}).to_csv(
    OUTPUT_DIR / "final_selected_genes.csv",
    index=False,
)


# ============================================================
# PREDICT UNKNOWN TEST CELLS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTING UNKNOWN NA TEST CELLS")
print("=" * 70)

test_prediction_info = frequency_weighted_predict(
    final_model,
    train_df,
    test_df,
    gene_columns,
    best_alpha,
)

test_output = test_df.copy()

test_output["normal_prediction"] = test_prediction_info["normal_prediction"]
test_output[TARGET] = test_prediction_info["weighted_prediction"]
test_output["prediction_changed"] = test_prediction_info["prediction_changed"]
test_output["normal_probability"] = test_prediction_info["normal_probability"]
test_output["model_probability_for_weighted_class"] = test_prediction_info["model_probability_for_weighted_class"]
test_output["training_frequency_prior"] = test_prediction_info["training_frequency_prior"]
test_output["weighted_probability"] = test_prediction_info["weighted_probability"]
test_output["prior_source"] = test_prediction_info["prior_source"]
test_output["chosen_model"] = best_model_name
test_output["chosen_number_of_genes"] = best_k
test_output["frequency_alpha"] = best_alpha

print(f"\nTest predictions changed by frequency weighting: {test_output['prediction_changed'].sum():,} / {len(test_output):,}")


# ============================================================
# SAVE FULL TEST OUTPUT
# ============================================================

test_output.drop(
    columns=["_dataset_key", "_ap_key"],
    errors="ignore",
).to_csv(
    OUTPUT_DIR / "NA_test_frequency_weighted_predictions.csv",
    index=False,
)


# ============================================================
# SIMPLE TEST PREDICTION FILE
# ============================================================

simple_test_output = pd.DataFrame({
    "identity_number": test_output[TEST_ID],
    "predicted_cell_type": test_output[TARGET],
})

simple_test_output.to_csv(
    OUTPUT_DIR / "NA_test_predictions_simple.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame({
    "metric": [
        "selected_model",
        "selected_number_of_genes",
        "selected_alpha",
        "normal_accuracy",
        "weighted_accuracy",
        "normal_balanced_accuracy",
        "weighted_balanced_accuracy",
        "normal_macro_f1",
        "weighted_macro_f1",
        "number_predictions_changed",
    ],

    "value": [
        best_model_name,
        best_k,
        best_alpha,
        normal_accuracy,
        weighted_accuracy,
        normal_balanced,
        weighted_balanced,
        normal_f1,
        weighted_f1,
        len(changed_df),
    ],
})

summary_df.to_csv(
    OUTPUT_DIR / "frequency_weighting_summary.csv",
    index=False,
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"\nBest model: {best_model_name}")
print(f"Best genes: {best_k}")
print(f"Best frequency alpha: {best_alpha}")

print(f"\nNormal held-out accuracy: {normal_accuracy:.3%}")
print(f"Weighted held-out accuracy: {weighted_accuracy:.3%}")

print(f"\nNormal held-out balanced accuracy: {normal_balanced:.3%}")
print(f"Weighted held-out balanced accuracy: {weighted_balanced:.3%}")

print(f"\nResults saved in:\n{OUTPUT_DIR}")

print("\nImportant files:")
print("  frequency_weighted_model_comparison.csv")
print("  frequency_weighting_summary.csv")
print("  training_frequency_priors.csv")
print("  best_model_identity_predictions.csv")
print("  best_model_identity_predictions_simple.csv")
print("  predictions_changed_by_frequency.csv")
print("  incorrect_weighted_predictions.csv")
print("  accuracy_by_cell_type.csv")
print("  weighted_confusion_matrix.csv")
print("  final_selected_genes.csv")
print("  NA_test_frequency_weighted_predictions.csv")
print("  NA_test_predictions_simple.csv")