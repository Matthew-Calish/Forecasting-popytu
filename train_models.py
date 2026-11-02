"""
train_models.py: Modele bazowe, trening LightGBM w wariantach A, B, C oraz ewaluacja i wizualizacja wyników na zbiorze walidacyjnym i testowym.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, # MAE - o ile średnio mylimy się w prognozie dziennego popytu
    root_mean_squared_error, #RMSE - metryka mocniej karząca za duże błędy w nietypowe dni
    mean_absolute_percentage_error, # MAPE - średni błąd procentowy
    r2_score, # R^2 
)

import project_config as cfg
import common_fun as cf

sns.set_theme(style="whitegrid", palette="deep")


def ensure_dirs():
    Path(cfg.MODEL_PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def prepare_categories(df: pd.DataFrame) -> pd.DataFrame:

    """Upewnia się, że kolumny kategoryczne mają stałe kategorie z project_config.py."""

    df = df.copy()
    df["season"] = pd.Categorical(df["season"], categories=cfg.SEASONS)
    df["weather"] = pd.Categorical(df["weather"], categories=cfg.WHEATER)
    return df



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
            "Liczba_cech": 1,
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


def train_and_evaluate(df: pd.DataFrame | None = None, save: bool = True):

    ensure_dirs()
    if df is None:
        df = cf.load_model_data()
    df = prepare_categories(df)

    train, val, test = cf.split_by_date(df)
    train_val = pd.concat([train, val], axis=0).reset_index(drop=True)

    # Tabela z modelami bazowymi (Etap 2.1)
    baseline_df = evaluate_baselines(df)
    rows = baseline_df.to_dict("records")

    val_preds = {cfg.DATE: val[cfg.DATE].to_numpy(),
                 cfg.TARGET: val[cfg.TARGET].to_numpy()}

    test_preds = {
        cfg.DATE: test[cfg.DATE].to_numpy(),
        cfg.TARGET: test[cfg.TARGET].to_numpy(),
        "weather": test["weather"],
        "temp_c": test["temp_c"].to_numpy(),
        "humidity": test["humidity"].to_numpy(),
        "is_holiday": test["is_holiday"].to_numpy(),
    }

    for model_name, col in cfg.BASELINES.items():
        val_preds[model_name] = val[col].to_numpy()
        test_preds[model_name] = test[col].to_numpy()

    trained_models = {}

    for var_key, model_name in cfg.VARIANT_LABELS.items():
        X_train, y_train = cf.get_xy(train, var_key)
        X_val, y_val = cf.get_xy(val, var_key)
        X_train_val, y_train_val = cf.get_xy(train_val, var_key)
        X_test, y_test = cf.get_xy(test, var_key)

        # Krok 1: Trening na train z early stopping na val
        model_val = lgb.LGBMRegressor(**cfg.LGBM_PARAMS)
        model_val.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )

        best_iter = model_val.best_iteration_
        pred_val = model_val.predict(X_val)
        m_val = compute_metrics(y_val, pred_val)
        val_preds[model_name] = pred_val

        # Krok 2: Refit na (train + val) przed ewaluacją na test
        refit_params = {**cfg.LGBM_PARAMS, "n_estimators": int(best_iter * 1.15)}
        model_final = lgb.LGBMRegressor(**refit_params)
        model_final.fit(X_train_val, y_train_val)

        pred_test = model_final.predict(X_test)
        m_test = compute_metrics(y_test, pred_test)
        test_preds[model_name] = pred_test

        trained_models[var_key] = {
            "model_val": model_val,
            "model_final": model_final,
            "best_iter": best_iter,
            "X_train_val": X_train_val,
            "X_test": X_test,
            "y_test": y_test,
            "pred_test": pred_test,
        }

        rows.append({
            "Model": model_name,
            "Wariant": var_key,
            "Liczba_cech": len(cfg.VARIANTS[var_key]),
            "Val_MAE": m_val["MAE"],
            "Val_RMSE": m_val["RMSE"],
            "Val_MAPE_%": m_val["MAPE_%"],
            "Val_R2": m_val["R2"],
            "Test_MAE": m_test["MAE"],
            "Test_RMSE": m_test["RMSE"],
            "Test_MAPE_%": m_test["MAPE_%"],
            "Test_R2": m_test["R2"],
        })

    metrics_df = pd.DataFrame(rows)

    # Obliczenie redukcji błędu MAE względem Wariantu A i najlepszego Baseline'u na zbiorze Test
    mae_a = metrics_df.loc[metrics_df["Wariant"] == "A", "Test_MAE"].iloc[0]
    mae_base = metrics_df.loc[metrics_df["Wariant"] == "Baseline", "Test_MAE"].min()

    metrics_df["Poprawa_vs_Baseline_%"] = ((mae_base - metrics_df["Test_MAE"]) / mae_base * 100).round(2)
    metrics_df["Poprawa_vs_Wariant_A_%"] = ((mae_a - metrics_df["Test_MAE"]) / mae_a * 100).round(2)

    test_preds_df = pd.DataFrame(test_preds)
    val_preds_df = pd.DataFrame(val_preds)

    if save:
        res_dir = Path(cfg.RESULTS_DIR)
        metrics_df.to_parquet(res_dir / "metrics_comparison.parquet", index=False)
        test_preds_df.to_parquet(res_dir / "test_predictions.parquet", index=False)
        val_preds_df.to_parquet(res_dir / "val_predictions.parquet", index=False)

    return metrics_df, test_preds_df, trained_models


def plot_ablation_metrics(metrics_df: pd.DataFrame, save: bool = True):

    """Wykres 1: Porównanie błędów MAE i RMSE na zbiorze testowym (Baseline vs Warianty A, B, C)."""

    ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    palette = ["#95a5a6", "#95a5a6", "#7f8c8d", "#3498db", "#2ecc71", "#16a085"]

    sns.barplot(
        data=metrics_df, y="Model", x="Test_MAE", ax=axes[0],
        palette=palette, hue="Model", legend=False
    )
    for i, v in enumerate(metrics_df["Test_MAE"]):
        axes[0].text(v + 15, i, f"{v:.1f}", va="center", fontweight="bold", fontsize=9.5)
    axes[0].set_title("Błąd MAE na zbiorze testowym", fontweight="bold")
    axes[0].set_xlabel("MAE [liczba wypożyczeń]")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, metrics_df["Test_MAE"].max() * 1.18)

    sns.barplot(
        data=metrics_df, y="Model", x="Test_RMSE", ax=axes[1],
        palette=palette, hue="Model", legend=False
    )
    for i, v in enumerate(metrics_df["Test_RMSE"]):
        axes[1].text(v + 18, i, f"{v:.1f}", va="center", fontweight="bold", fontsize=9.5)
    axes[1].set_title("Błąd RMSE na zbiorze testowym", fontweight="bold")
    axes[1].set_xlabel("RMSE [liczba wypożyczeń]")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, metrics_df["Test_RMSE"].max() * 1.18)

    fig.suptitle("Analiza ablacji: wpływ zmiennych zewnętrznych (Wariant B i C) na dokładność prognozy", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.MODEL_PLOTS_DIR) / "01_ablation_mae_rmse.png", dpi=150, bbox_inches="tight")
    return fig


def plot_forecast_vs_actual(test_preds_df: pd.DataFrame, save: bool = True):

    """Wykres 2: Przebieg prognozy Wariantu A vs Wariantu B/C na tle rzeczywistego popytu w zbiorze testowym."""

    ensure_dirs()
    col_a = cfg.VARIANT_LABELS["A"]
    col_b = cfg.VARIANT_LABELS["B"]
    col_c = cfg.VARIANT_LABELS["C"]

    mae_b = mean_absolute_error(test_preds_df[cfg.TARGET], test_preds_df[col_b])
    mae_c = mean_absolute_error(test_preds_df[cfg.TARGET], test_preds_df[col_c])
    best_weather_col = col_c if mae_c <= mae_b else col_b

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(
        test_preds_df[cfg.DATE], test_preds_df[cfg.TARGET],
        label="Rzeczywisty popyt (demand)", color="black", lw=2.0, alpha=0.85
    )
    ax.plot(
        test_preds_df[cfg.DATE], test_preds_df[col_a],
        label=f"{col_a}", color="#e74c3c", linestyle="--", lw=1.5, alpha=0.85
    )
    ax.plot(
        test_preds_df[cfg.DATE], test_preds_df[best_weather_col],
        label=f"{best_weather_col}", color="#27ae60", lw=1.8, alpha=0.95
    )

    bad_days = test_preds_df[test_preds_df["weather"] == "Bad"]
    ax.scatter(
        bad_days[cfg.DATE], bad_days[cfg.TARGET],
        color="#2980b9", s=65, zorder=5, label="Dni ze złą pogodą / opadami (weather = Bad)"
    )

    ax.set_title("Prognoza na zbiorze testowym (Jesień–Zima 2012): Wariant A (bez pogody) vs Model z pogodą", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel("Dzienny popyt")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.MODEL_PLOTS_DIR) / "02_forecast_vs_actual_test.png", dpi=150)
    return fig


def plot_residuals_by_weather(test_preds_df: pd.DataFrame, save: bool = True):

    """Wykres 3: Porównanie bezwzględnych błędów predykcji Wariantu A vs B vs C w podziale na pogodę."""

    ensure_dirs()
    col_a = cfg.VARIANT_LABELS["A"]
    col_b = cfg.VARIANT_LABELS["B"]
    col_c = cfg.VARIANT_LABELS["C"]

    records = []
    for _, row in test_preds_df.iterrows():
        for label, col in [
            ("Wariant A (bez pogody)", col_a),
            ("Wariant B (+ pogoda)", col_b),
            ("Wariant C (+ cechy pochodne)", col_c),
        ]:
            records.append({
                "Warunki pogodowe": row["weather"],
                "Wariant": label,
                "Błąd bezwzględny |y - y_hat|": abs(row[cfg.TARGET] - row[col]),
            })
    err_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.barplot(
        data=err_df, x="Warunki pogodowe", y="Błąd bezwzględny |y - y_hat|",
        hue="Wariant", order=cfg.WHEATER, errorbar=None, palette=["#e74c3c", "#3498db", "#2ecc71"], ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9, fontweight="bold")

    ax.set_title("Średni błąd bezwzględny (MAE) w zależności od warunków pogodowych na zbiorze testowym", fontsize=12, fontweight="bold")
    ax.set_xlabel("Warunki pogodowe (weather)")
    ax.set_ylabel("Średni błąd bezwzględny MAE [liczba wypożyczeń]")
    ax.legend(title="Model")
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.MODEL_PLOTS_DIR) / "03_residuals_comparison.png", dpi=150)
    return fig


def run_all_training(save: bool = True):

    """Uruchamia pełny Etap 2 (Baseline + LightGBM A/B/C + wykresy ablacji)."""
    
    metrics_df, test_preds_df, trained_models = train_and_evaluate(save=save)

    print("=== ETAP 2.2: PORÓWNANIE MODELI I WARIANTÓW CECH (ABLACJA A vs B vs C) ===")
    print(metrics_df.to_string(index=False))

    fig1 = plot_ablation_metrics(metrics_df, save=save)
    fig2 = plot_forecast_vs_actual(test_preds_df, save=save)
    fig3 = plot_residuals_by_weather(test_preds_df, save=save)

    if save:
        print(f"\nZapisano tabele wyników (.parquet) w: {Path(cfg.RESULTS_DIR).resolve()}")
        print(f"Zapisano wykresy ewaluacji (.png) w: {Path(cfg.MODEL_PLOTS_DIR).resolve()}")

    return metrics_df, test_preds_df, trained_models, (fig1, fig2, fig3)


if __name__ == "__main__":
    run_all_training(save=True)
    plt.close("all")
