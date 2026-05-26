"""
Reconstructed from predict_crop_phenotype.so (Python 3.10, Cython).
Confidence notes:
  - All model names, variable names, pipeline structure, metric names: HIGH
  - Numeric values in param grids (logspace bounds/steps, n_splits, etc.): UNCERTAIN
    — filled with typical values used in genomic prediction literature
  - y_AC variable: uncertain meaning; reconstructed as y_actual (actual labels)
  - Order of pipelines list: uncertain; Ridge/RR-BLUP listed first as it's
    the traditional genomic-prediction baseline
"""

import os
import math

import numpy as np
import pandas as pd
from configparser import ConfigParser
from scipy.stats import pearsonr

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ── Scoring ──────────────────────────────────────────────────────────────────

def pearson_correlation_scorer(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return r if not math.isnan(r) else 0.0


pcc_scorer = make_scorer(pearson_correlation_scorer, greater_is_better=True)
mae_scorer = make_scorer(mean_absolute_error,        greater_is_better=False)


# ── Param grids ──────────────────────────────────────────────────────────────
# Exact logspace bounds are unknown; these are standard starting points.

params_rr = {
    "pca__n_components": [50, 100, 200],
    "model__alpha": np.logspace(-4, 4, 9).tolist(),
}

params_svr = {
    "pca__n_components": [50, 100, 200],
    "model__C":      np.logspace(-2, 3, 6).tolist(),
    "model__gamma":  np.logspace(-4, 1, 6).tolist(),
    "model__kernel": ["rbf"],
}

params_rf = {
    "pca__n_components":    [50, 100, 200],
    "model__n_estimators":  [100, 200, 500],
    "model__max_depth":     [3, 5, 10, None],
    "model__min_samples_leaf": [1, 2, 4],
}

params_gbr = {
    "pca__n_components":       [50, 100, 200],
    "model__n_estimators":     [100, 200, 500],
    "model__learning_rate":    [0.01, 0.05, 0.1],
    "model__max_depth":        [3, 5],
    "model__min_samples_leaf": [1, 2, 4],
}

params_pls = {
    "n_components": list(range(2, 21)),
}


# ── Public entry point ───────────────────────────────────────────────────────

def predict(trait_names):
    """
    Run grid-searched cross-validation for Ridge, SVR, RF, GBR, and PLS on
    each trait and print the best PCC and MAE.

    Parameters
    ----------
    trait_names : list of str
        Column names in the CROP_VECTOR_TRAIT CSV to use as targets.
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.ini")
    cf = ConfigParser()
    cf.read(file_path, encoding="utf-8")

    vector_trait_path = cf.get("path", "CROP_VECTOR_TRAIT")

    if not os.path.exists(vector_trait_path):
        raise FileNotFoundError(f"CROP_VECTOR_TRAIT file not found: {vector_trait_path}")

    df = pd.read_csv(vector_trait_path, encoding="utf-8")
    df = df.dropna(subset=trait_names, how="any")
    df = df.reset_index(drop=True)

    # Feature columns = everything that isn't a trait column
    feature_cols = [c for c in df.columns if c not in trait_names]
    X = df[feature_cols].values

    n_splits = 5
    cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pipelines for sklearn-based models (Ridge, SVR, RF, GBR)
    pipe_rr  = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("model", Ridge())])
    pipe_svr = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("model", SVR())])
    pipe_rf  = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("model", RandomForestRegressor(random_state=42))])
    pipe_gbr = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("model", GradientBoostingRegressor(random_state=42))])

    pipelines = [
        ("RR-BLUP/Ridge",     pipe_rr,  params_rr),
        ("SVR",               pipe_svr, params_svr),
        ("Random Forest",     pipe_rf,  params_rf),
        ("Gradient Boosting", pipe_gbr, params_gbr),
    ]

    scoring = {
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "PCC": pcc_scorer,
    }

    results = []

    for target_trait_name in trait_names:
        targets   = df[target_trait_name].values
        target_df = pd.DataFrame({"Target": targets})

        print(f"\n{'='*60}")
        print(f"Trait: {target_trait_name}")

        best_pcc = -np.inf
        best_mae = np.inf
        best_params = None
        model_name = None

        for name, pipeline, param_grid in pipelines:
            grid_search = GridSearchCV(
                estimator  = pipeline,
                param_grid = param_grid,
                scoring    = scoring,
                refit      = "PCC",
                cv         = cv_strategy,
                n_jobs     = -1,
                verbose    = 0,
            )
            grid_search.fit(X, targets)

            cv_results  = grid_search.cv_results_
            best_index_ = grid_search.best_index_
            best_pcc_   = cv_results["mean_test_PCC"][best_index_]
            best_mae_   = -cv_results["mean_test_MAE"][best_index_]   # stored as negative
            best_params_= grid_search.best_params_

            print(f"\n  Model: {name}")
            print(f"  Best Params: {best_params_}")
            print(f"  Best PCC: {best_pcc_:.4f}")
            print(f"  Corresponding MAE: {best_mae_:.4f}")

            if best_pcc_ > best_pcc:
                best_pcc    = best_pcc_
                best_mae    = best_mae_
                best_params = best_params_
                model_name  = name

        # ── PLS (separate GridSearch — no PCA/scaler prefix needed) ──
        pls_model = PLSRegression()
        grid_search_pls = GridSearchCV(
            estimator  = pls_model,
            param_grid = params_pls,
            scoring    = scoring,
            refit      = "PCC",
            cv         = cv_strategy,
            n_jobs     = -1,
            verbose    = 0,
        )
        grid_search_pls.fit(X, targets)

        cv_results   = grid_search_pls.cv_results_
        best_index_  = grid_search_pls.best_index_
        best_pcc_pls = cv_results["mean_test_PCC"][best_index_]
        best_mae_pls = -cv_results["mean_test_MAE"][best_index_]

        print(f"\n  Model: PLS")
        print(f"  Best Params: {grid_search_pls.best_params_}")
        print(f"  Best PCC: {best_pcc_pls:.4f}")
        print(f"  MAE: {best_mae_pls:.4f}")

        if best_pcc_pls > best_pcc:
            best_pcc    = best_pcc_pls
            best_mae    = best_mae_pls
            best_params = grid_search_pls.best_params_
            model_name  = "PLS"

        # ── Summary ──────────────────────────────────────────────────
        y_AC = targets   # y_actual (y_AC symbol from source)
        results.append({
            "Model":      model_name,
            "Trait":      target_trait_name,
            "PCC":        round(best_pcc, 4),
            "MAE":        round(best_mae, 4),
            "BestParams": best_params,
        })

        print(f"\n  >> Best overall for {target_trait_name}:")
        print(f"     Model: {model_name}")
        print(f"     PCC:   {best_pcc:.4f}")
        print(f"     MAE:   {best_mae:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="PCC", ascending=False).reset_index(drop=True)
    print("\n\nFinal results:")
    print(results_df.to_string())

    return results_df
