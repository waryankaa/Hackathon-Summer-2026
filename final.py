from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate


# ============================================================
# FILES
# ============================================================

ROOT = Path("/Users/abigailwaryanka/Desktop/Hackathon-Summer-2026")

TRAIN_FILE = ROOT / "data/full_train.csv"
TEST_FILE = ROOT / "data/full_test.csv"

PAIR_FILE = ROOT / "difficult_cell_types_nearest_neighbor_genes (1)(1).csv"

OUT = ROOT / "final_combined_predictions"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "MERFISH_cell_type_annotation"
SECTION_COL = "Segment"
EI_COL = "Excitatory_vs_Inhibitory"
SEED = 42

MODELS = [
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
]

GENE_COUNTS = [1, 2, 3, 5, 10, 20, 40]

INNER_MAX_SPLITS = 5

NUMERIC_METADATA = [
    "AP_position",
    "volume",
]

NON_GENE = [
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
# BEST NA MODEL SETTINGS
# ============================================================

ASTRO_T = 0.70
OLIGO_T = 0.30

# Highest-accuracy oligo specialist settings
M1 = 0.25
M2 = 0.25

ASTRO = [
    "astrocyte_1",
    "astrocyte_2",
]

OLIGO = [
    "oligodendrocyte_1",
    "oligodendrocyte_2",
    "oligodendrocyte_precursor_cell",
    "oligodendrocyte_progenitor_1",
    "oligodendrocyte_progenitor_2",
]

PAIR1 = tuple(sorted([
    "oligodendrocyte_1",
    "oligodendrocyte_progenitor_2",
]))

PAIR2 = tuple(sorted([
    "oligodendrocyte_precursor_cell",
    "oligodendrocyte_progenitor_1",
]))


# ============================================================
# GENERAL HELPERS
# ============================================================

class SafeSelectKBest(BaseEstimator, TransformerMixin):

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):

        X = np.asarray(X)

        if X.shape[1] == 0:
            raise ValueError("No features remain.")

        self.k_ = min(
            self.k,
            X.shape[1],
        )

        scores, _ = f_classif(
            X,
            y,
        )

        scores = np.nan_to_num(
            scores,
            nan=-np.inf,
            posinf=np.finfo(float).max,
            neginf=-np.inf,
        )

        self.selected_indices_ = np.sort(
            np.argsort(scores)[::-1][:self.k_]
        )

        return self

    def transform(self, X):

        return np.asarray(X)[
            :, self.selected_indices_
        ]


def log1p_nonnegative(X):

    return np.log1p(
        np.clip(
            np.asarray(X, dtype=float),
            0,
            None,
        )
    )


def clean_group(x):

    if pd.isna(x):
        return "__NA__"

    try:

        n = float(x)

        if n.is_integer():
            return str(int(n))

    except (ValueError, TypeError):
        pass

    return str(x)


def prepare(df):

    df = df.copy()

    df["_section"] = (
        df[SECTION_COL]
        .apply(clean_group)
    )

    df["_ei"] = (
        df[EI_COL]
        .apply(clean_group)
    )

    return df


def id_column(df):

    first = df.columns[0]

    if (
        str(first).startswith("Unnamed:")
        or df[first].nunique() == len(df)
    ):
        return first

    return None


def gene_columns(train, test, id_col):

    excluded = set(
        NON_GENE
        + ["_section", "_ei"]
    )

    if id_col is not None:
        excluded.add(id_col)

    return [
        c
        for c in train.columns
        if c not in excluded
        and c in test.columns
    ]


# ============================================================
# ORIGINAL HIERARCHICAL MODEL
# ============================================================

def build_pipeline(
    model_name,
    k,
    genes,
    numeric,
):

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value=0,
            ),
        ),
        (
            "log",
            FunctionTransformer(
                log1p_nonnegative,
                validate=False,
            ),
        ),
        (
            "variance",
            VarianceThreshold(0),
        ),
        (
            "select",
            SafeSelectKBest(k),
        ),
    ]

    if model_name == "LogisticRegression":

        steps.append(
            (
                "scale",
                StandardScaler(),
            )
        )

    transformers = [
        (
            "genes",
            Pipeline(steps),
            genes,
        )
    ]

    if numeric:

        transformers.append(
            (
                "metadata",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
                    (
                        "scale",
                        StandardScaler(),
                    ),
                ]),
                numeric,
            )
        )

    pre = ColumnTransformer(
        transformers,
        remainder="drop",
    )

    if model_name == "LogisticRegression":

        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        )

    elif model_name == "RandomForest":

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

    else:

        clf = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

    return Pipeline([
        (
            "preprocessor",
            pre,
        ),
        (
            "classifier",
            clf,
        ),
    ])


# ============================================================
# TRAIN MODEL FOR AMBIGUOUS HIERARCHICAL GROUP
# ============================================================

def fit_ambiguous(
    group,
    genes,
    numeric,
):

    y = (
        group[TARGET]
        .astype(str)
    )

    smallest = int(
        y.value_counts().min()
    )

    # Too few examples for CV
    if smallest < 2:

        k = min(
            5,
            len(genes),
        )

        if k < 1:
            return None, None

        m = build_pipeline(
            "ExtraTrees",
            k,
            genes,
            numeric,
        )

        try:

            m.fit(
                group,
                y,
            )

            return m, {
                "model": "ExtraTrees",
                "genes": k,
            }

        except Exception:

            return None, None

    cv = StratifiedKFold(
        n_splits=min(
            INNER_MAX_SPLITS,
            smallest,
        ),
        shuffle=True,
        random_state=SEED,
    )

    results = []

    for name in MODELS:

        for k in [
            x
            for x in GENE_COUNTS
            if x <= len(genes)
        ]:

            m = build_pipeline(
                name,
                k,
                genes,
                numeric,
            )

            try:

                s = cross_validate(
                    m,
                    group,
                    y,
                    cv=cv,
                    scoring={
                        "accuracy": "accuracy",
                        "balanced": "balanced_accuracy",
                        "f1": "f1_macro",
                    },
                    n_jobs=-1,
                    error_score="raise",
                )

            except Exception:

                continue

            results.append([
                name,
                k,
                s["test_accuracy"].mean(),
                s["test_balanced"].mean(),
                s["test_f1"].mean(),
            ])

    if not results:

        return None, None

    r = pd.DataFrame(
        results,
        columns=[
            "model",
            "genes",
            "accuracy",
            "balanced",
            "macro_f1",
        ],
    ).sort_values(
        [
            "balanced",
            "macro_f1",
            "genes",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    best = r.iloc[0]

    m = build_pipeline(
        best.model,
        int(best.genes),
        genes,
        numeric,
    )

    m.fit(
        group,
        y,
    )

    return m, {
        "model": best.model,
        "genes": int(best.genes),
    }


# ============================================================
# BUILD HIERARCHICAL ROUTER
# ============================================================

def fit_router(
    train,
    genes,
    numeric,
):

    router = {

        "direct_section": {},
        "direct_ei": {},
        "models": {},
        "model_info": {},
        "section_majority": {},

        "global_majority": (
            train[TARGET]
            .mode()
            .iloc[0]
        ),
    }

    rows = []

    for section, sdf in train.groupby(
        "_section"
    ):

        counts = (
            sdf[TARGET]
            .value_counts()
        )

        router[
            "section_majority"
        ][section] = counts.idxmax()

        # Entire section = one cell type
        if len(counts) == 1:

            router[
                "direct_section"
            ][section] = counts.index[0]

            rows.append([
                section,
                "ALL",
                "direct_section",
                counts.index[0],
                "",
                np.nan,
            ])

            continue

        # Otherwise split by E/I/NA
        for ei, edf in sdf.groupby(
            "_ei"
        ):

            labels = (
                edf[TARGET]
                .value_counts()
            )

            key = (
                section,
                ei,
            )

            # One cell type in this subgroup
            if len(labels) == 1:

                router[
                    "direct_ei"
                ][key] = labels.index[0]

                rows.append([
                    section,
                    ei,
                    "direct_section_EI",
                    labels.index[0],
                    "",
                    np.nan,
                ])

                continue

            # Need classifier
            m, info = fit_ambiguous(
                edf,
                genes,
                numeric,
            )

            if m is None:

                rows.append([
                    section,
                    ei,
                    "section_majority_fallback",
                    "",
                    "",
                    np.nan,
                ])

                continue

            router[
                "models"
            ][key] = m

            router[
                "model_info"
            ][key] = info

            rows.append([
                section,
                ei,
                "subgroup_model",
                " | ".join(
                    sorted(
                        labels.index.astype(str)
                    )
                ),
                info["model"],
                info["genes"],
            ])

    routing = pd.DataFrame(
        rows,
        columns=[
            "section",
            "ei_group",
            "route_type",
            "cell_types",
            "model",
            "number_of_genes",
        ],
    )

    return (
        router,
        routing,
    )


# ============================================================
# PREDICT WITH HIERARCHICAL ROUTER
# ============================================================

def predict_router(
    router,
    data,
):

    pred = []
    method = []
    model_name = []
    k = []

    for _, row in data.iterrows():

        section = row["_section"]
        ei = row["_ei"]

        key = (
            section,
            ei,
        )

        if section in router[
            "direct_section"
        ]:

            p = router[
                "direct_section"
            ][section]

            meth = "direct_section"
            m = ""
            n = np.nan

        elif key in router[
            "direct_ei"
        ]:

            p = router[
                "direct_ei"
            ][key]

            meth = "direct_section_EI"
            m = ""
            n = np.nan

        elif key in router[
            "models"
        ]:

            p = router[
                "models"
            ][key].predict(
                pd.DataFrame([row])
            )[0]

            info = router[
                "model_info"
            ][key]

            meth = "subgroup_model"
            m = info["model"]
            n = info["genes"]

        elif section in router[
            "section_majority"
        ]:

            p = router[
                "section_majority"
            ][section]

            meth = (
                "section_majority_fallback"
            )

            m = ""
            n = np.nan

        else:

            p = router[
                "global_majority"
            ]

            meth = (
                "global_majority_fallback"
            )

            m = ""
            n = np.nan

        pred.append(p)
        method.append(meth)
        model_name.append(m)
        k.append(n)

    return pd.DataFrame(
        {
            "prediction": pred,
            "prediction_method": method,
            "chosen_model": model_name,
            "chosen_number_of_genes": k,
        },
        index=data.index,
    )


# ============================================================
# IDENTIFY NA GROUP
# ============================================================

def is_na_group(s):

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "__na__",
            "na",
            "nan",
            "none",
            "",
        ])
    )


# ============================================================
# NA GENE FEATURES
# ============================================================

def log_genes(
    df,
    genes,
):

    return np.log1p(
        df[genes]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .astype(float)
    )


def na_features(
    df,
    genes,
    volume_median,
):

    X = log_genes(
        df,
        genes,
    ).copy()

    X["volume"] = (
        pd.to_numeric(
            df["volume"],
            errors="coerce",
        )
        .fillna(volume_median)
        .to_numpy()
    )

    return X


# ============================================================
# CALIBRATED NA EXTRATREES
# ============================================================

def calibrated_forest(y):

    smallest = int(
        pd.Series(y)
        .value_counts()
        .min()
    )

    cv = min(
        3,
        smallest,
    )

    if cv < 2:

        raise ValueError(
            "Not enough cells to calibrate NA classifier."
        )

    base = ExtraTreesClassifier(
        n_estimators=1000,
        class_weight="balanced",
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )

    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=cv,
        n_jobs=-1,
    )


# ============================================================
# FREQUENCY CORRECTION
# ============================================================

def expected_counts(
    freq,
    n,
):

    x = freq * n

    c = np.floor(
        x
    ).astype(int)

    for i in np.argsort(
        -(x - c)
    )[
        : n - c.sum()
    ]:

        c[i] += 1

    return c


def constrained_assign(
    P,
    labels,
    c,
):

    slots = np.repeat(
        np.arange(
            len(labels)
        ),
        c,
    )

    r, s = linear_sum_assignment(
        -np.log(
            np.clip(
                P[:, slots],
                1e-12,
                1,
            )
        )
    )

    out = np.empty(
        len(P),
        dtype=object,
    )

    out[r] = np.array(
        labels
    )[slots[s]]

    return out


def frequency_correct(
    out,
    P,
    classes,
    y,
    members,
    pos,
):

    members = [
        m
        for m in members
        if m in classes
    ]

    if (
        len(pos) == 0
        or len(members) < 2
    ):
        return

    cols = [
        np.where(
            classes == m
        )[0][0]
        for m in members
    ]

    freq = (
        y[
            y.isin(members)
        ]
        .value_counts(
            normalize=True
        )
    )

    c = expected_counts(
        np.array([
            freq.get(
                m,
                0,
            )
            for m in members
        ]),
        len(pos),
    )

    out[pos] = constrained_assign(
        P[pos][:, cols],
        members,
        c,
    )


# ============================================================
# LOAD PAIR-SPECIFIC OLIGO GENES
# ============================================================

def read_pair_genes(genes):

    if PAIR_FILE.exists():

        file = PAIR_FILE

    else:

        hits = list(
            ROOT.rglob(
                "difficult_cell_types_nearest_neighbor_genes*.csv"
            )
        )

        if not hits:

            raise FileNotFoundError(
                "Could not find difficult-cell-type gene CSV."
            )

        file = hits[0]

    pg = pd.read_csv(file)

    pg["FDR"] = pd.to_numeric(
        pg["FDR"],
        errors="coerce",
    )

    pg = pg[
        (pg["FDR"] < 0.05)
        & pg["gene"].isin(genes)
    ].copy()

    pg["pair"] = pg.apply(
        lambda r: tuple(
            sorted([
                str(
                    r["cell_type"]
                ),
                str(
                    r["nearest_cell_type"]
                ),
            ])
        ),
        axis=1,
    )

    return {

        pair: (
            pg.loc[
                pg["pair"] == pair
            ]
            .sort_values("FDR")[
                "gene"
            ]
            .drop_duplicates()
            .tolist()
        )

        for pair in [
            PAIR1,
            PAIR2,
        ]
    }


# ============================================================
# TRAIN OLIGO SPECIALISTS
# ============================================================

def train_oligo_specialists(
    train,
    genes,
    pair_genes,
):

    G = log_genes(
        train,
        genes,
    )

    specialists = {}

    for pair, margin in [
        (
            PAIR1,
            M1,
        ),
        (
            PAIR2,
            M2,
        ),
    ]:

        gs = pair_genes.get(
            pair,
            [],
        )

        idx = train.index[
            train[TARGET]
            .astype(str)
            .isin(pair)
        ]

        if (
            not gs
            or train.loc[
                idx,
                TARGET,
            ].nunique() != 2
        ):
            continue

        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            ),
        )

        m.fit(
            G.loc[
                idx,
                gs,
            ],
            train.loc[
                idx,
                TARGET,
            ].astype(str),
        )

        specialists[
            pair
        ] = (
            m,
            gs,
            margin,
        )

    return specialists


# ============================================================
# TRAIN + PREDICT NA CELLS
# ============================================================

def predict_na(
    na_train,
    na_test,
    genes,
):

    y = (
        na_train[TARGET]
        .astype(str)
    )

    volume_median = (
        pd.to_numeric(
            na_train["volume"],
            errors="coerce",
        )
        .median()
    )

    print(
        "\nTraining calibrated NA ExtraTrees..."
    )

    model = calibrated_forest(
        y
    )

    model.fit(
        na_features(
            na_train,
            genes,
            volume_median,
        ),
        y,
    )

    P = model.predict_proba(
        na_features(
            na_test,
            genes,
            volume_median,
        )
    )

    classes = model.classes_

    raw = classes[
        P.argmax(axis=1)
    ]

    confidence = P.max(
        axis=1
    )

    # ========================================================
    # FREQUENCY CORRECTION
    # ========================================================

    out = raw.copy()

    frequency_correct(
        out,
        P,
        classes,
        y,
        ASTRO,
        np.where(
            np.isin(
                raw,
                ASTRO,
            )
            & (
                confidence
                < ASTRO_T
            )
        )[0],
    )

    frequency_correct(
        out,
        P,
        classes,
        y,
        OLIGO,
        np.where(
            np.isin(
                raw,
                OLIGO,
            )
            & (
                confidence
                < OLIGO_T
            )
        )[0],
    )

    # ========================================================
    # OLIGO SPECIALISTS
    # ========================================================

    pair_genes = read_pair_genes(
        genes
    )

    print(
        "\nOligodendrocyte specialist genes:"
    )

    for pair, gs in pair_genes.items():

        print(
            f"{pair}: {gs}"
        )

    specialists = (
        train_oligo_specialists(
            na_train,
            genes,
            pair_genes,
        )
    )

    Gtest = log_genes(
        na_test,
        genes,
    )

    top = np.argsort(
        P,
        axis=1,
    )[
        :, -2:
    ][
        :, ::-1
    ]

    specialist_changes = 0

    for i, (
        a,
        b,
    ) in enumerate(top):

        pair = tuple(
            sorted([
                classes[a],
                classes[b],
            ])
        )

        if pair not in specialists:

            continue

        m, gs, margin = (
            specialists[pair]
        )

        probability_gap = (
            P[i, a]
            - P[i, b]
        )

        if (
            probability_gap
            > margin
        ):

            continue

        if out[i] not in pair:

            continue

        old = out[i]

        out[i] = m.predict(
            Gtest.iloc[
                [i]
            ][gs]
        )[0]

        specialist_changes += (
            old != out[i]
        )

    final_probability = np.array([
        P[
            i,
            np.where(
                classes == label
            )[0][0],
        ]

        for i, label
        in enumerate(out)
    ])

    print(
        f"NA specialist label changes: "
        f"{specialist_changes:,}"
    )

    return (
        out,
        raw,
        confidence,
        final_probability,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOAD DATA"
    )

    print(
        "=" * 70
    )

    train = pd.read_csv(
        TRAIN_FILE
    )

    test = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Training cells: "
        f"{len(train):,}"
    )

    print(
        f"Test cells:     "
        f"{len(test):,}"
    )

    if TARGET not in train.columns:

        raise ValueError(
            f"{TARGET} missing from training data."
        )

    for col in [
        SECTION_COL,
        EI_COL,
    ]:

        if (
            col not in train.columns
            or col not in test.columns
        ):

            raise ValueError(
                f"{col} must exist in train and test."
            )

    # Remove target from unknown test data if present
    if TARGET in test.columns:

        test = test.drop(
            columns=[
                TARGET
            ]
        )

    train = prepare(
        train
    )

    test = prepare(
        test
    )

    id_col = id_column(
        train
    )

    genes = gene_columns(
        train,
        test,
        id_col,
    )

    print(
        f"Cell ID column: "
        f"{id_col}"
    )

    print(
        f"Shared genes:   "
        f"{len(genes)}"
    )

    # Numeric genes
    for g in genes:

        train[g] = pd.to_numeric(
            train[g],
            errors="coerce",
        ).fillna(0)

        test[g] = pd.to_numeric(
            test[g],
            errors="coerce",
        ).fillna(0)

    numeric = [
        c
        for c in NUMERIC_METADATA
        if c in train.columns
        and c in test.columns
    ]

    for c in numeric:

        train[c] = pd.to_numeric(
            train[c],
            errors="coerce",
        )

        test[c] = pd.to_numeric(
            test[c],
            errors="coerce",
        )

    # ========================================================
    # TRAIN HIERARCHICAL MODEL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING HIERARCHICAL MODEL"
    )

    print(
        "=" * 70
    )

    router, routing = fit_router(
        train,
        genes,
        numeric,
    )

    routing.to_csv(
        OUT
        / "hierarchical_routing.csv",
        index=False,
    )

    prediction_info = predict_router(
        router,
        test,
    )

    # ========================================================
    # IDENTIFY NA CELLS
    # ========================================================

    train_na = is_na_group(
        train["_ei"]
    )

    test_na = is_na_group(
        test["_ei"]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NA MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"NA training cells: "
        f"{train_na.sum():,}"
    )

    print(
        f"NA test cells:     "
        f"{test_na.sum():,}"
    )

    raw_na = np.array(
        [],
        dtype=object,
    )

    conf_na = np.array(
        [],
        dtype=float,
    )

    final_prob = np.array(
        [],
        dtype=float,
    )

    # ========================================================
    # BEST NA MODEL OVERRIDE
    # ========================================================

    if (
        train_na.sum() > 0
        and test_na.sum() > 0
    ):

        (
            final_na,
            raw_na,
            conf_na,
            final_prob,
        ) = predict_na(
            train.loc[
                train_na
            ].copy(),
            test.loc[
                test_na
            ].copy(),
            genes,
        )

        prediction_info.loc[
            test_na,
            "prediction",
        ] = final_na

        prediction_info.loc[
            test_na,
            "prediction_method",
        ] = (
            "calibrated_NA_best_accuracy"
        )

        prediction_info.loc[
            test_na,
            "chosen_model",
        ] = (
            "CalibratedExtraTrees"
            "+frequency"
            "+oligo_specialists"
        )

        prediction_info.loc[
            test_na,
            "chosen_number_of_genes",
        ] = len(genes)

    # ========================================================
    # FULL AUDIT FILE
    # ========================================================

    audit = test.copy()

    audit[TARGET] = (
        prediction_info[
            "prediction"
        ]
    )

    audit[
        "prediction_method"
    ] = prediction_info[
        "prediction_method"
    ]

    audit[
        "chosen_model"
    ] = prediction_info[
        "chosen_model"
    ]

    audit[
        "chosen_number_of_genes"
    ] = prediction_info[
        "chosen_number_of_genes"
    ]

    # ========================================================
    # IMPORTANT FIX:
    # raw_na_prediction must be object/string dtype
    # ========================================================

    audit[
        "raw_na_prediction"
    ] = pd.Series(
        pd.NA,
        index=audit.index,
        dtype="object",
    )

    audit[
        "raw_na_confidence"
    ] = np.nan

    audit[
        "final_na_base_probability"
    ] = np.nan

    if test_na.sum() > 0:

        audit.loc[
            test_na,
            "raw_na_prediction",
        ] = np.asarray(
            raw_na,
            dtype=object,
        )

        audit.loc[
            test_na,
            "raw_na_confidence",
        ] = np.asarray(
            conf_na,
            dtype=float,
        )

        audit.loc[
            test_na,
            "final_na_base_probability",
        ] = np.asarray(
            final_prob,
            dtype=float,
        )

    audit_out = audit.drop(
        columns=[
            "_section",
            "_ei",
        ],
        errors="ignore",
    )

    audit_out.to_csv(
        OUT
        / "test_predictions_with_details.csv",
        index=False,
    )

    # ========================================================
    # FINAL SUBMISSION
    # ========================================================

    if (
        id_col is not None
        and id_col in audit.columns
    ):

        submission = audit[
            [
                id_col,
                TARGET,
            ]
        ].copy()

    else:

        submission = audit[
            [
                TARGET,
            ]
        ].copy()

    submission.to_csv(
        OUT
        / "prediction.csv",
        index=False,
    )

    # ========================================================
    # TRAIN VS TEST FREQUENCIES
    # ========================================================

    train_freq = (
        train[TARGET]
        .astype(str)
        .value_counts(
            normalize=True
        )
        .rename(
            "train_frequency"
        )
    )

    predicted_freq = (
        submission[TARGET]
        .astype(str)
        .value_counts(
            normalize=True
        )
        .rename(
            "predicted_test_frequency"
        )
    )

    frequency_comparison = pd.concat(
        [
            train_freq,
            predicted_freq,
        ],
        axis=1,
    ).fillna(0)

    frequency_comparison[
        "difference"
    ] = (
        frequency_comparison[
            "predicted_test_frequency"
        ]
        - frequency_comparison[
            "train_frequency"
        ]
    )

    frequency_comparison.to_csv(
        OUT
        / "class_frequency_comparison.csv"
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        "\nPrediction methods:"
    )

    print(
        audit[
            "prediction_method"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinal predicted cell types:"
    )

    print(
        submission[
            TARGET
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTrain vs predicted-test frequencies:"
    )

    print(
        frequency_comparison
        .round(4)
        .to_string()
    )

    print(
        "\nOUTPUT FILES"
    )

    print(
        f"Final prediction: "
        f"{OUT/'prediction.csv'}"
    )

    print(
        f"Detailed audit:   "
        f"{OUT/'test_predictions_with_details.csv'}"
    )

    print(
        f"Frequencies:      "
        f"{OUT/'class_frequency_comparison.csv'}"
    )

    print(
        f"Routing:          "
        f"{OUT/'hierarchical_routing.csv'}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()