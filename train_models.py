"""
train_models.py: Modele bazowe (Baseline), trening LightGBM w wariantach A, B, C (Ablacja)
oraz ewaluacja i wizualizacja wyników na zbiorze walidacyjnym i testowym.
"""

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, # MAE - o ile średnio mylimy się w prognozie dziennego popytu
    root_mean_squared_error, #RMSE - metryka mocniej karząca za duże błędy w nietypowe dni
    mean_absolute_percentage_error, # MAPE - średni błąd procentowy
    r2_score, # R^2 
)

import project_config as cfg
import common_fun as cf


def compute_metrics(y_true, y_pred) -> dict:
    """Oblicza wspólny zestaw metryk błędu prognozy (MAE, RMSE, MAPE [%], R2)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE_%": round(float(mape), 2),
        "R2": round(float(r2), 4),
    }


def evaluate_baselines(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Oblicza metryki na zbiorze walidacyjnym (val) i testowym (test)
    dla modeli bazowych zdefiniowanych w project_config.py (BASELINES).
    """
    if df is None:
        df = cf.load_model_data()

    _, val, test = cf.split_by_date(df)

    rows = []
    for model_name, col in cfg.BASELINES.items():
        m_val = compute_metrics(val[cfg.TARGET], val[col])
        m_test = compute_metrics(test[cfg.TARGET], test[col])

        rows.append({
            "Model": model_name,
            "Wariant": "Baseline",
            "Cecha": col,
            "Val_MAE": m_val["MAE"],
            "Val_RMSE": m_val["RMSE"],
            "Val_MAPE_%": m_val["MAPE_%"],
            "Val_R2": m_val["R2"],
            "Test_MAE": m_test["MAE"],
            "Test_RMSE": m_test["RMSE"],
            "Test_MAPE_%": m_test["MAPE_%"],
            "Test_R2": m_test["R2"],
        })

    return pd.DataFrame(rows)


def run_baselines():
    """Uruchamia ewaluację modeli bazowych (Etap 2.1) i wyświetla tabelę wyników."""
    baseline_df = evaluate_baselines()
    print("=== WYNIKI MODELI BAZOWYCH ===")
    print(baseline_df.to_string(index=False))
    return baseline_df


if __name__ == "__main__":
    run_baselines()
