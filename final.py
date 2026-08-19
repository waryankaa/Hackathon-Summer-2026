from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = Path("/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/data/full_train.csv")
TEST_FILE = Path("/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/data/full_test.csv")
OUTPUT_DIR = Path("/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026/hierarchical_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "MERFISH_cell_type_annotation"

# Change to "Section_ID" if you mean Section_ID instead of Segment.
SECTION_COL = "Segment"

EI_COL = "Excitatory_vs_Inhibitory"
MOUSE_GROUP_COL = "Mouse_ID"

RANDOM_STATE = 42

OUTER_SPLITS = 5
INNER_MAX_SPLITS = 5

# "stratified" = random cells
# "mouse" = hold out entire mice
OUTER_MODE = "stratified"

MODEL_NAMES = ["LogisticRegression", "RandomForest", "ExtraTrees"]
GENE_COUNTS = [1, 2, 3, 5, 10, 20, 40]

NUMERIC_METADATA = ["AP_position", "volume"]

NON_GENE_COLUMNS = [
    "Datasets",
    "volume",
    "center_x",
    "center_y",
    TARGET,
    "Region",
    EI_COL,
    "Segment",
    "Gender",
    "Mouse_ID",
    "AP_position",
    "Section_ID",
]


# ============================================================
# SAFE FEATURE SELECTION
# ============================================================

class SafeSelectKBest(BaseEstimator, TransformerMixin):

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)

        if X.shape[1] == 0:
            raise ValueError("No features remain after variance filtering.")

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
        X = np.asarray(X)
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        if indices:
            return self.selected_indices_

        mask = np.zeros(len(self.scores_), dtype=bool)
        mask[self.selected_indices_] = True
        return mask


# ============================================================
# DATA HELPERS
# ============================================================

def log1p_nonnegative(X):
    X = np.asarray(X, dtype=float)
    return np.log1p(np.clip(X, 0, None))


def clean_group_value(value):

    if pd.isna(value):
        return "__NA__"

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

    except (ValueError, TypeError):
        pass

    return str(value)


def prepare_dataframe(df):
    df = df.copy()
    df["_section"] = df[SECTION_COL].apply(clean_group_value)
    df["_ei"] = df[EI_COL].apply(clean_group_value)
    return df


def find_id_column(df):

    first_column = df.columns[0]

    if str(first_column).startswith("Unnamed:") or df[first_column].nunique() == len(df):
        return first_column

    return None


def find_gene_columns(train_df, test_df, id_column):

    excluded = set(NON_GENE_COLUMNS + ["_section", "_ei"])

    if id_column is not None:
        excluded.add(id_column)

    return [
        column
        for column in train_df.columns
        if column not in excluded and column in test_df.columns
    ]


# ============================================================
# BUILD MODEL
# ============================================================

def build_pipeline(model_name, number_of_genes, gene_columns, numeric_metadata):

    gene_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("log_transform", FunctionTransformer(log1p_nonnegative, validate=False)),
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        ("select_genes", SafeSelectKBest(k=number_of_genes)),
    ]

    if model_name == "LogisticRegression":
        gene_steps.append(("scale_genes", StandardScaler()))

    gene_pipeline = Pipeline(gene_steps)

    transformers = [
        ("genes", gene_pipeline, gene_columns),
    ]

    if numeric_metadata:

        metadata_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(("metadata", metadata_pipeline, numeric_metadata))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "LogisticRegression":

        classifier = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    elif model_name == "RandomForest":

        classifier = RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    elif model_name == "ExtraTrees":

        classifier = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ============================================================
# CHOOSE MODEL FOR ONE AMBIGUOUS GROUP
# ============================================================

def fit_ambiguous_model(group_df, gene_columns, numeric_metadata):

    y = group_df[TARGET]
    class_counts = y.value_counts()
    smallest_class = int(class_counts.min())

    # Not enough examples for proper inner CV.
    if smallest_class < 2:

        default_k = min(5, len(gene_columns))

        if default_k < 1:
            return None, None, pd.DataFrame()

        try:

            model = build_pipeline(
                "ExtraTrees",
                default_k,
                gene_columns,
                numeric_metadata,
            )

            model.fit(group_df, y)

            model_info = {
                "model": "ExtraTrees",
                "number_of_genes": default_k,
                "inner_accuracy": np.nan,
                "inner_balanced_accuracy": np.nan,
                "inner_macro_f1": np.nan,
                "selection_status": "low_data_default_model",
            }

            return model, model_info, pd.DataFrame()

        except Exception:
            return None, None, pd.DataFrame()

    n_splits = min(INNER_MAX_SPLITS, smallest_class)

    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    gene_counts = [k for k in GENE_COUNTS if k <= len(gene_columns)]
    results = []

    for model_name in MODEL_NAMES:

        for k in gene_counts:

            pipeline = build_pipeline(
                model_name,
                k,
                gene_columns,
                numeric_metadata,
            )

            try:

                scores = cross_validate(
                    pipeline,
                    group_df,
                    y,
                    cv=inner_cv,
                    scoring={
                        "accuracy": "accuracy",
                        "balanced_accuracy": "balanced_accuracy",
                        "macro_f1": "f1_macro",
                    },
                    n_jobs=-1,
                    error_score="raise",
                )

            except Exception:
                continue

            results.append({
                "model": model_name,
                "number_of_genes": k,
                "accuracy": scores["test_accuracy"].mean(),
                "accuracy_sd": scores["test_accuracy"].std(),
                "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
                "balanced_accuracy_sd": scores["test_balanced_accuracy"].std(),
                "macro_f1": scores["test_macro_f1"].mean(),
                "macro_f1_sd": scores["test_macro_f1"].std(),
            })

    if not results:
        return None, None, pd.DataFrame()

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["balanced_accuracy", "macro_f1", "number_of_genes"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best = results_df.iloc[0]

    best_model = build_pipeline(
        best["model"],
        int(best["number_of_genes"]),
        gene_columns,
        numeric_metadata,
    )

    best_model.fit(group_df, y)

    model_info = {
        "model": best["model"],
        "number_of_genes": int(best["number_of_genes"]),
        "inner_accuracy": best["accuracy"],
        "inner_balanced_accuracy": best["balanced_accuracy"],
        "inner_macro_f1": best["macro_f1"],
        "selection_status": "cross_validated",
    }

    return best_model, model_info, results_df


# ============================================================
# BUILD HIERARCHICAL ROUTER
# ============================================================

def fit_router(train_df, gene_columns, numeric_metadata):

    router = {
        "direct_section": {},
        "direct_ei": {},
        "models": {},
        "model_info": {},
        "section_majority": {},
        "global_majority": train_df[TARGET].mode().iloc[0],
    }

    routing_records = []
    all_model_results = []

    for section, section_df in train_df.groupby("_section"):

        section_counts = section_df[TARGET].value_counts()
        section_labels = section_counts.index.tolist()

        router["section_majority"][section] = section_counts.idxmax()

        # Entire section contains only one cell type.
        if len(section_labels) == 1:

            label = section_labels[0]
            router["direct_section"][section] = label

            routing_records.append({
                "section": section,
                "ei_group": "ALL",
                "n_cells": len(section_df),
                "n_cell_types": 1,
                "cell_types": label,
                "route_type": "direct_section",
                "best_model": "",
                "number_of_genes": np.nan,
                "inner_accuracy": np.nan,
                "inner_balanced_accuracy": np.nan,
                "inner_macro_f1": np.nan,
            })

            continue

        # Section contains multiple labels, so split by E/I.
        for ei_value, ei_df in section_df.groupby("_ei"):

            ei_counts = ei_df[TARGET].value_counts()
            ei_labels = ei_counts.index.tolist()

            key = (section, ei_value)

            # E/I subgroup contains only one type.
            if len(ei_labels) == 1:

                label = ei_labels[0]
                router["direct_ei"][key] = label

                routing_records.append({
                    "section": section,
                    "ei_group": ei_value,
                    "n_cells": len(ei_df),
                    "n_cell_types": 1,
                    "cell_types": label,
                    "route_type": "direct_section_EI",
                    "best_model": "",
                    "number_of_genes": np.nan,
                    "inner_accuracy": np.nan,
                    "inner_balanced_accuracy": np.nan,
                    "inner_macro_f1": np.nan,
                })

                continue

            # Still ambiguous: model is required.
            model, model_info, model_results = fit_ambiguous_model(
                ei_df,
                gene_columns,
                numeric_metadata,
            )

            cell_type_string = " | ".join(sorted(map(str, ei_labels)))

            if model is None:

                routing_records.append({
                    "section": section,
                    "ei_group": ei_value,
                    "n_cells": len(ei_df),
                    "n_cell_types": len(ei_labels),
                    "cell_types": cell_type_string,
                    "route_type": "section_majority_fallback",
                    "best_model": "",
                    "number_of_genes": np.nan,
                    "inner_accuracy": np.nan,
                    "inner_balanced_accuracy": np.nan,
                    "inner_macro_f1": np.nan,
                })

                continue

            router["models"][key] = model
            router["model_info"][key] = model_info

            route_name = (
                "subgroup_model"
                if model_info["selection_status"] == "cross_validated"
                else "low_data_subgroup_model"
            )

            routing_records.append({
                "section": section,
                "ei_group": ei_value,
                "n_cells": len(ei_df),
                "n_cell_types": len(ei_labels),
                "cell_types": cell_type_string,
                "route_type": route_name,
                "best_model": model_info["model"],
                "number_of_genes": model_info["number_of_genes"],
                "inner_accuracy": model_info["inner_accuracy"],
                "inner_balanced_accuracy": model_info["inner_balanced_accuracy"],
                "inner_macro_f1": model_info["inner_macro_f1"],
            })

            if len(model_results) > 0:

                model_results = model_results.copy()
                model_results["section"] = section
                model_results["ei_group"] = ei_value

                all_model_results.append(model_results)

    routing_df = pd.DataFrame(routing_records)

    if all_model_results:
        model_results_df = pd.concat(all_model_results, ignore_index=True)
    else:
        model_results_df = pd.DataFrame()

    return router, routing_df, model_results_df


# ============================================================
# PREDICT USING ROUTER
# ============================================================

def predict_with_router(router, data_df):

    predictions = []
    methods = []
    chosen_models = []
    chosen_gene_counts = []

    for _, row in data_df.iterrows():

        section = row["_section"]
        ei_value = row["_ei"]
        key = (section, ei_value)

        if section in router["direct_section"]:

            prediction = router["direct_section"][section]
            method = "direct_section"
            model_name = ""
            gene_count = np.nan

        elif key in router["direct_ei"]:

            prediction = router["direct_ei"][key]
            method = "direct_section_EI"
            model_name = ""
            gene_count = np.nan

        elif key in router["models"]:

            model = router["models"][key]
            prediction = model.predict(pd.DataFrame([row]))[0]

            info = router["model_info"][key]

            method = (
                "subgroup_model"
                if info["selection_status"] == "cross_validated"
                else "low_data_subgroup_model"
            )

            model_name = info["model"]
            gene_count = info["number_of_genes"]

        elif section in router["section_majority"]:

            prediction = router["section_majority"][section]
            method = "section_majority_fallback"
            model_name = ""
            gene_count = np.nan

        else:

            prediction = router["global_majority"]
            method = "global_majority_fallback"
            model_name = ""
            gene_count = np.nan

        predictions.append(prediction)
        methods.append(method)
        chosen_models.append(model_name)
        chosen_gene_counts.append(gene_count)

    return pd.DataFrame({
        "prediction": predictions,
        "prediction_method": methods,
        "chosen_model": chosen_models,
        "chosen_number_of_genes": chosen_gene_counts,
    }, index=data_df.index)


# ============================================================
# OUTER CROSS-VALIDATION SPLITS
# ============================================================

def make_outer_splits(train_df):

    y = train_df[TARGET]

    if OUTER_MODE == "mouse":

        if MOUSE_GROUP_COL not in train_df.columns:
            raise ValueError(f"{MOUSE_GROUP_COL} was not found.")

        groups = train_df[MOUSE_GROUP_COL].fillna("__NA__").astype(str)
        n_splits = min(OUTER_SPLITS, groups.nunique())

        if n_splits < 2:
            raise ValueError("Not enough mice for grouped cross-validation.")

        splitter = GroupKFold(n_splits=n_splits)

        return (
            list(splitter.split(train_df, y, groups)),
            f"GroupKFold by {MOUSE_GROUP_COL} ({n_splits} folds)",
        )

    class_counts = y.value_counts()
    smallest_class = int(class_counts.min())

    if smallest_class >= 2:

        n_splits = min(OUTER_SPLITS, smallest_class)

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        return (
            list(splitter.split(train_df, y)),
            f"StratifiedKFold ({n_splits} folds)",
        )

    n_splits = min(OUTER_SPLITS, len(train_df))

    if n_splits < 2:
        raise ValueError("Not enough cells for cross-validation.")

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return (
        list(splitter.split(train_df)),
        f"KFold ({n_splits} folds; stratification impossible)",
    )


# ============================================================
# MODEL STABILITY ACROSS OUTER FOLDS
# ============================================================

def calculate_model_stability(routing_df):

    if len(routing_df) == 0:
        return pd.DataFrame()

    total_outer_folds = routing_df["outer_fold"].nunique()

    model_rows = routing_df[
        routing_df["route_type"].isin([
            "subgroup_model",
            "low_data_subgroup_model",
        ])
    ].copy()

    model_rows = model_rows[
        model_rows["best_model"].notna()
        & (model_rows["best_model"] != "")
    ].copy()

    if len(model_rows) == 0:
        return pd.DataFrame()

    model_rows["configuration"] = (
        model_rows["best_model"].astype(str)
        + " + "
        + model_rows["number_of_genes"].fillna(-1).astype(int).astype(str)
        + " genes"
    )

    stability_records = []

    for (section, ei_group), group in model_rows.groupby(["section", "ei_group"]):

        model_counts = group["best_model"].value_counts()
        configuration_counts = group["configuration"].value_counts()

        most_common_model = model_counts.index[0]
        model_wins = int(model_counts.iloc[0])

        most_common_configuration = configuration_counts.index[0]
        configuration_wins = int(configuration_counts.iloc[0])

        folds_with_model = group["outer_fold"].nunique()

        stability_records.append({
            "section": section,
            "ei_group": ei_group,
            "total_outer_folds": total_outer_folds,
            "folds_with_model": folds_with_model,
            "most_common_cv_model": most_common_model,
            "model_wins": model_wins,
            "model_stability": model_wins / total_outer_folds,
            "model_stability_when_modeled": model_wins / folds_with_model,
            "most_common_cv_configuration": most_common_configuration,
            "configuration_wins": configuration_wins,
            "configuration_stability": configuration_wins / total_outer_folds,
            "configuration_stability_when_modeled": configuration_wins / folds_with_model,
        })

    return pd.DataFrame(stability_records)


# ============================================================
# EVALUATE ENTIRE PIPELINE
# ============================================================

def evaluate_full_pipeline(train_df, gene_columns, numeric_metadata, id_column):

    print("\n" + "=" * 70)
    print("FULL PIPELINE CROSS-VALIDATION")
    print("=" * 70)

    outer_splits, split_description = make_outer_splits(train_df)

    print(f"\nOuter evaluation: {split_description}")

    all_predictions = []
    all_routing = []
    all_inner_results = []

    for fold_number, (train_index, validation_index) in enumerate(outer_splits, start=1):

        print("\n" + "-" * 70)
        print(f"OUTER FOLD {fold_number}")
        print("-" * 70)

        outer_train = train_df.iloc[train_index].copy()
        outer_validation = train_df.iloc[validation_index].copy()

        print(f"Training cells: {len(outer_train):,}")
        print(f"Validation cells: {len(outer_validation):,}")

        router, routing_df, inner_results_df = fit_router(
            outer_train,
            gene_columns,
            numeric_metadata,
        )

        prediction_info = predict_with_router(router, outer_validation)

        fold_predictions = outer_validation[
            [TARGET, "_section", "_ei"]
        ].copy()

        fold_predictions["original_row_index"] = outer_validation.index

        if id_column is not None and id_column in outer_validation.columns:
            fold_predictions["cell_id"] = outer_validation[id_column]

        fold_predictions["predicted_cell_type"] = prediction_info["prediction"]
        fold_predictions["prediction_method"] = prediction_info["prediction_method"]
        fold_predictions["chosen_model"] = prediction_info["chosen_model"]
        fold_predictions["chosen_number_of_genes"] = prediction_info["chosen_number_of_genes"]

        fold_predictions["correct"] = (
            fold_predictions[TARGET]
            == fold_predictions["predicted_cell_type"]
        )

        fold_predictions["outer_fold"] = fold_number

        all_predictions.append(fold_predictions)

        if len(routing_df) > 0:

            routing_df = routing_df.copy()
            routing_df["outer_fold"] = fold_number

            all_routing.append(routing_df)

        if len(inner_results_df) > 0:

            inner_results_df = inner_results_df.copy()
            inner_results_df["outer_fold"] = fold_number

            all_inner_results.append(inner_results_df)

        print(f"\nFold accuracy: {fold_predictions['correct'].mean():.3%}")

        print("\nPrediction methods:")
        print(fold_predictions["prediction_method"].value_counts().to_string())

    # ========================================================
    # COMBINE OUTER VALIDATION PREDICTIONS
    # ========================================================

    cv_predictions = pd.concat(all_predictions, ignore_index=True)

    y_true = cv_predictions[TARGET]
    y_pred = cv_predictions["predicted_cell_type"]

    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    overall_macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 70)
    print("OVERALL HIERARCHICAL CLASSIFIER PERFORMANCE")
    print("=" * 70)

    print(f"\nCells evaluated: {len(cv_predictions):,}")
    print(f"Overall accuracy:  {overall_accuracy:.3%}")
    print(f"Balanced accuracy: {overall_balanced_accuracy:.3%}")
    print(f"Macro F1:          {overall_macro_f1:.3%}")

    # ========================================================
    # PERFORMANCE BY ROUTING METHOD
    # ========================================================

    method_accuracy = (
        cv_predictions
        .groupby("prediction_method")
        .agg(
            cells=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values("cells", ascending=False)
    )

    print("\n" + "-" * 70)
    print("ACCURACY BY PREDICTION METHOD")
    print("-" * 70)

    print(method_accuracy.to_string(index=False))

    # ========================================================
    # PERFORMANCE BY CELL TYPE
    # ========================================================

    cell_type_accuracy = (
        cv_predictions
        .groupby(TARGET)
        .agg(
            cells=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values("accuracy")
    )

    print("\n" + "-" * 70)
    print("HARDEST CELL TYPES")
    print("-" * 70)

    print(cell_type_accuracy.head(20).to_string(index=False))

    # ========================================================
    # PERFORMANCE BY SECTION
    # ========================================================

    section_accuracy = (
        cv_predictions
        .groupby("_section")
        .agg(
            cells=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values("accuracy")
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    labels = sorted(y_true.unique().tolist())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[f"Actual_{label}" for label in labels],
        columns=[f"Predicted_{label}" for label in labels],
    )

    # ========================================================
    # SAVE GENERAL CV RESULTS
    # ========================================================

    cv_predictions.to_csv(
        OUTPUT_DIR / "overall_cv_predictions.csv",
        index=False,
    )

    method_accuracy.to_csv(
        OUTPUT_DIR / "accuracy_by_prediction_method.csv",
        index=False,
    )

    cell_type_accuracy.to_csv(
        OUTPUT_DIR / "accuracy_by_cell_type.csv",
        index=False,
    )

    section_accuracy.to_csv(
        OUTPUT_DIR / "accuracy_by_section.csv",
        index=False,
    )

    confusion_df.to_csv(
        OUTPUT_DIR / "overall_confusion_matrix.csv"
    )

    summary_df = pd.DataFrame({
        "metric": [
            "overall_accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
        "value": [
            overall_accuracy,
            overall_balanced_accuracy,
            overall_macro_f1,
        ],
    })

    summary_df.to_csv(
        OUTPUT_DIR / "overall_cv_summary.csv",
        index=False,
    )

    # ========================================================
    # ROUTING + MODEL STABILITY
    # ========================================================

    if all_routing:

        outer_routing_df = pd.concat(
            all_routing,
            ignore_index=True,
        )

        outer_routing_df.to_csv(
            OUTPUT_DIR / "outer_cv_routing.csv",
            index=False,
        )

        model_stability_df = calculate_model_stability(
            outer_routing_df
        )

        model_stability_df.to_csv(
            OUTPUT_DIR / "model_stability_across_outer_folds.csv",
            index=False,
        )

        print("\n" + "-" * 70)
        print("MODEL STABILITY ACROSS OUTER FOLDS")
        print("-" * 70)

        if len(model_stability_df) > 0:

            display_stability = model_stability_df.copy()

            for column in [
                "model_stability",
                "model_stability_when_modeled",
                "configuration_stability",
                "configuration_stability_when_modeled",
            ]:
                display_stability[column] = (
                    display_stability[column] * 100
                ).round(1)

            print(display_stability.to_string(index=False))

    else:
        model_stability_df = pd.DataFrame()

    if all_inner_results:

        pd.concat(
            all_inner_results,
            ignore_index=True,
        ).to_csv(
            OUTPUT_DIR / "outer_cv_inner_model_results.csv",
            index=False,
        )

    return {
        "accuracy": overall_accuracy,
        "balanced_accuracy": overall_balanced_accuracy,
        "macro_f1": overall_macro_f1,
        "model_stability": model_stability_df,
    }


# ============================================================
# TRAIN FINAL SYSTEM + PREDICT TEST DATA
# ============================================================

def fit_final_and_predict(
    train_df,
    test_df,
    gene_columns,
    numeric_metadata,
    id_column,
    model_stability_df,
):

    print("\n" + "=" * 70)
    print("TRAINING FINAL SYSTEM ON ALL LABELED DATA")
    print("=" * 70)

    final_router, final_routing_df, final_model_results = fit_router(
        train_df,
        gene_columns,
        numeric_metadata,
    )

    # ========================================================
    # COMPARE FINAL MODEL TO OUTER-CV CONSENSUS
    # ========================================================

    if len(model_stability_df) > 0:

        final_comparison = final_routing_df.merge(
            model_stability_df,
            on=["section", "ei_group"],
            how="left",
        )

        final_comparison["final_configuration"] = np.where(
            final_comparison["best_model"].fillna("") != "",
            final_comparison["best_model"].astype(str)
            + " + "
            + final_comparison["number_of_genes"].fillna(-1).astype(int).astype(str)
            + " genes",
            "",
        )

        final_comparison["final_model_matches_cv_mode"] = (
            final_comparison["best_model"]
            == final_comparison["most_common_cv_model"]
        )

        final_comparison["final_configuration_matches_cv_mode"] = (
            final_comparison["final_configuration"]
            == final_comparison["most_common_cv_configuration"]
        )

        final_comparison.to_csv(
            OUTPUT_DIR / "final_vs_cv_model_stability.csv",
            index=False,
        )

        modeled_comparison = final_comparison[
            final_comparison["best_model"].fillna("") != ""
        ].copy()

        print("\n" + "-" * 70)
        print("FINAL MODEL VS OUTER-CV CONSENSUS")
        print("-" * 70)

        if len(modeled_comparison) > 0:

            columns = [
                "section",
                "ei_group",
                "best_model",
                "number_of_genes",
                "most_common_cv_model",
                "model_stability",
                "most_common_cv_configuration",
                "configuration_stability",
                "final_model_matches_cv_mode",
                "final_configuration_matches_cv_mode",
            ]

            print(modeled_comparison[columns].to_string(index=False))

    else:

        final_comparison = final_routing_df.copy()

        final_comparison.to_csv(
            OUTPUT_DIR / "final_vs_cv_model_stability.csv",
            index=False,
        )

    # ========================================================
    # SHOW FINAL ROUTING
    # ========================================================

    print("\nFinal routing structure:\n")

    if len(final_routing_df) > 0:

        columns = [
            "section",
            "ei_group",
            "n_cells",
            "n_cell_types",
            "cell_types",
            "route_type",
            "best_model",
            "number_of_genes",
            "inner_balanced_accuracy",
        ]

        print(final_routing_df[columns].to_string(index=False))

    # ========================================================
    # PREDICT UNKNOWN TEST SET
    # ========================================================

    prediction_info = predict_with_router(
        final_router,
        test_df,
    )

    test_output = test_df.copy()

    test_output[TARGET] = prediction_info["prediction"]
    test_output["prediction_method"] = prediction_info["prediction_method"]
    test_output["chosen_model"] = prediction_info["chosen_model"]
    test_output["chosen_number_of_genes"] = prediction_info["chosen_number_of_genes"]

    print("\n" + "-" * 70)
    print("UNKNOWN TEST SET PREDICTION METHODS")
    print("-" * 70)

    print(test_output["prediction_method"].value_counts().to_string())

    # ========================================================
    # SAVE AUDIT FILE
    # ========================================================

    audit_output = test_output.drop(
        columns=["_section", "_ei"],
        errors="ignore",
    )

    audit_output.to_csv(
        OUTPUT_DIR / "test_predictions_with_methods.csv",
        index=False,
    )

    # ========================================================
    # SUBMISSION FILE
    # ========================================================

    if id_column is not None and id_column in test_output.columns:
        submission = test_output[[id_column, TARGET]].copy()
    else:
        submission = test_output[[TARGET]].copy()

    submission.to_csv(
        OUTPUT_DIR / "prediction.csv",
        index=False,
    )

    # ========================================================
    # SAVE FINAL ROUTING + MODEL RESULTS
    # ========================================================

    final_routing_df.to_csv(
        OUTPUT_DIR / "final_routing_summary.csv",
        index=False,
    )

    if len(final_model_results) > 0:

        final_model_results.to_csv(
            OUTPUT_DIR / "final_model_results_by_group.csv",
            index=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("\n" + "=" * 70)
    print("DATA")
    print("=" * 70)

    print(f"\nLabeled training cells: {len(train_df):,}")
    print(f"Unknown test cells: {len(test_df):,}")

    if TARGET not in train_df.columns:
        raise ValueError(f"{TARGET} is missing from training data.")

    for column in [SECTION_COL, EI_COL]:

        if column not in train_df.columns or column not in test_df.columns:
            raise ValueError(f"{column} must exist in both train and test data.")

    train_df = prepare_dataframe(train_df)
    test_df = prepare_dataframe(test_df)

    id_column = find_id_column(train_df)

    print(f"\nCell ID column: {id_column}")

    gene_columns = find_gene_columns(
        train_df,
        test_df,
        id_column,
    )

    print(f"Gene columns shared between train and test: {len(gene_columns):,}")

    for gene in gene_columns:

        train_df[gene] = pd.to_numeric(
            train_df[gene],
            errors="coerce",
        ).fillna(0)

        test_df[gene] = pd.to_numeric(
            test_df[gene],
            errors="coerce",
        ).fillna(0)

    numeric_metadata = [
        column
        for column in NUMERIC_METADATA
        if column in train_df.columns and column in test_df.columns
    ]

    print(f"Numeric metadata used: {numeric_metadata}")

    for column in numeric_metadata:

        train_df[column] = pd.to_numeric(
            train_df[column],
            errors="coerce",
        )

        test_df[column] = pd.to_numeric(
            test_df[column],
            errors="coerce",
        )

    # ========================================================
    # STEP 1: CROSS-VALIDATE THE ENTIRE HIERARCHICAL SYSTEM
    # ========================================================

    evaluation_results = evaluate_full_pipeline(
        train_df,
        gene_columns,
        numeric_metadata,
        id_column,
    )

    # ========================================================
    # STEP 2: TRAIN FINAL SYSTEM ON ALL LABELED DATA
    # ========================================================

    fit_final_and_predict(
        train_df,
        test_df,
        gene_columns,
        numeric_metadata,
        id_column,
        evaluation_results["model_stability"],
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(f"\nCross-validated overall accuracy: {evaluation_results['accuracy']:.3%}")
    print(f"Cross-validated balanced accuracy: {evaluation_results['balanced_accuracy']:.3%}")
    print(f"Cross-validated macro F1: {evaluation_results['macro_f1']:.3%}")

    print(f"\nResults saved in:\n{OUTPUT_DIR}")

    print("\nImportant output files:")
    print("  overall_cv_summary.csv")
    print("  overall_cv_predictions.csv")
    print("  accuracy_by_prediction_method.csv")
    print("  accuracy_by_cell_type.csv")
    print("  accuracy_by_section.csv")
    print("  overall_confusion_matrix.csv")
    print("  outer_cv_routing.csv")
    print("  outer_cv_inner_model_results.csv")
    print("  model_stability_across_outer_folds.csv")
    print("  final_vs_cv_model_stability.csv")
    print("  final_routing_summary.csv")
    print("  final_model_results_by_group.csv")
    print("  test_predictions_with_methods.csv")
    print("  prediction.csv")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()