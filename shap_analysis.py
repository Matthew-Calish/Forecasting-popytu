"""
shap_analysis.py: Interpretowalność SHAP oraz ilościowa i wizualna analiza wpływu zmiennych zewnętrznych na prognozy dziennego popytu.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import project_config as cfg
import common_fun as cf
import train_models as tm

sns.set_theme(style="whitegrid", palette="deep")


def ensure_dirs():
    Path(cfg.SHAP_PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def compute_shap_values(variant: str = "C"):

    """
    Trenuje model LightGBM dla wskazanego wariantu (domyślnie najlepszy Wariant C)
    i oblicza wartości SHAP na zbiorze testowym za pomocą shap.TreeExplainer.
    """

    ensure_dirs()
    df = tm.prepare_categories(cf.load_model_data())
    _, _, test = cf.split_by_date(df)

    _, _, trained_models = tm.train_and_evaluate(df, save=False)
    model_info = trained_models[variant]
    model = model_info["model_final"]
    X_test = model_info["X_test"].reset_index(drop=True)
    test_reset = test.reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)

    return explanation, explainer, model, X_test, test_reset


def build_shap_impact_table(explanation, X_test: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Tworzy tabelę wpływu poszczególnych cech oraz podsumowanie grupowe: Historia/Kalendarz vs Zmienne zewnętrzne.
    """

    ensure_dirs()
    shap_vals = explanation.values
    mean_abs = np.abs(shap_vals).mean(axis=0)
    total_abs = mean_abs.sum()

    rows = []
    for idx, col in enumerate(X_test.columns):
        is_ext = col in cfg.EXTERNAL_COLS
        rows.append({
            "Cecha": col,
            "Grupa": "Zmienna zewnętrzna (B/C)" if is_ext else "Historia i kalendarz (A)",
            "Mean_Abs_SHAP": round(float(mean_abs[idx]), 2),
            "Udzial_%": round(float(mean_abs[idx] / total_abs * 100.0), 2),
            "Min_SHAP_korekta": round(float(shap_vals[:, idx].min()), 1),
            "Max_SHAP_korekta": round(float(shap_vals[:, idx].max()), 1),
        })

    impact_df = pd.DataFrame(rows).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    group_summary = (
        impact_df.groupby("Grupa", as_index=False)
        .agg(
            Liczba_cech=("Cecha", "count"),
            Suma_Mean_Abs_SHAP=("Mean_Abs_SHAP", "sum"),
            Laczny_Udzial_procent=("Udzial_%", "sum"),
        )
        .sort_values("Laczny_Udzial_procent", ascending=False)
        .reset_index(drop=True)
    )
    group_summary["Suma_Mean_Abs_SHAP"] = group_summary["Suma_Mean_Abs_SHAP"].round(2)
    group_summary["Laczny_Udzial_procent"] = group_summary["Laczny_Udzial_procent"].round(2)

    if save:
        res_dir = Path(cfg.RESULTS_DIR)
        impact_df.to_parquet(res_dir / "shap_external_impact.parquet", index=False)
        group_summary.to_parquet(res_dir / "shap_group_summary.parquet", index=False)

    return impact_df, group_summary


def plot_shap_beeswarm(explanation, X_test: pd.DataFrame, save: bool = True):

    """Wykres 1: Globalny rozkład wpływu cech SHAP (Beeswarm Summary Plot)."""

    ensure_dirs()
    fig = plt.figure(figsize=(11, 7.5))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title(
        "SHAP Summary Plot (Wariant C): kierunek i siła wpływu cech na prognozę popytu",
        fontsize=12.5, fontweight="bold", pad=14
    )
    plt.xlabel("Wartość SHAP (wpływ na prognozę dziennego popytu [liczba wypożyczeń])")
    plt.tight_layout()

    if save:
        fig.savefig(Path(cfg.SHAP_PLOTS_DIR) / "01_shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    return fig


def plot_shap_importance_bar(impact_df: pd.DataFrame, group_summary: pd.DataFrame, save: bool = True):

    """Wykres 2: Ranking ważności cech wg Mean |SHAP| z wyróżnieniem zmiennych zewnętrznych."""

    ensure_dirs()
    fig, ax = plt.subplots(figsize=(11.5, 7))

    palette = {
        "Historia i kalendarz (A)": "#3498db",
        "Zmienna zewnętrzna (B/C)": "#e67e22",
    }

    sns.barplot(
        data=impact_df, y="Cecha", x="Mean_Abs_SHAP",
        hue="Grupa", dodge=False, palette=palette, ax=ax
    )

    for i, row in impact_df.iterrows():
        ax.text(
            row["Mean_Abs_SHAP"] + 6, i,
            f"{row['Mean_Abs_SHAP']:.1f} ({row['Udzial_%']:.1f}%)",
            va="center", fontsize=8.8, fontweight="bold"
        )

    ext_share = group_summary.loc[
        group_summary["Grupa"] == "Zmienna zewnętrzna (B/C)", "Laczny_Udzial_procent"
    ].iloc[0]

    ax.set_title(
        f"Średni bezwzględny wpływ cech (Mean |SHAP|) – zmienne zewnętrzne odpowiadają za {ext_share:.1f}% wpływu",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Średnia zmiana prognozy |SHAP| [liczba wypożyczeń]")
    ax.set_ylabel("Cecha modelu")
    ax.set_xlim(0, impact_df["Mean_Abs_SHAP"].max() * 1.2)
    ax.legend(title="Kategoria cechy", loc="lower right", frameon=True)
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.SHAP_PLOTS_DIR) / "02_shap_importance_bar.png", dpi=150, bbox_inches="tight")
    return fig


def plot_shap_weather_dependence(explanation, X_test: pd.DataFrame, test_df: pd.DataFrame, save: bool = True):

    """Wykres 3: Wykresy zależności SHAP (Dependence Plots) dla kluczowych zmiennych pogodowych."""

    ensure_dirs()
    shap_vals = explanation.values
    cols = list(X_test.columns)

    weather_features = [
        ("temp_c", "Temperatura [°C]"),
        ("humidity", "Wilgotność względna [%]"),
        ("wind", "Prędkość wiatru [km/h]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=False)

    for ax, (feat, xlabel) in zip(axes, weather_features):
        idx = cols.index(feat)
        y_shap = shap_vals[:, idx]
        x_val = X_test[feat].to_numpy()

        sns.scatterplot(
            x=x_val, y=y_shap, hue=test_df["weather"],
            hue_order=cfg.WHEATER, palette="Set1", s=55, alpha=0.85, ax=ax
        )
        ax.axhline(0, color="black", linestyle="--", lw=1.1)

        # Gładka linia trendu (lowess / wielomian st. 2)
        valid = ~np.isnan(x_val)
        if valid.sum() > 5:
            poly = np.poly1d(np.polyfit(x_val[valid], y_shap[valid], deg=2))
            xs = np.linspace(np.nanmin(x_val), np.nanmax(x_val), 100)
            ax.plot(xs, poly(xs), color="#2c3e50", lw=1.8, linestyle="-", label="Trend wpływu SHAP")

        ax.set_title(f"Wpływ SHAP cechy: {feat}", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Korekta prognozy SHAP [wypożyczenia]")

    fig.suptitle(
        "Jak zmienne pogodowe korygują prognozę modelu LightGBM (SHAP Dependence Plots)",
        fontsize=13, fontweight="bold", y=1.03
    )
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.SHAP_PLOTS_DIR) / "03_shap_dependence_weather.png", dpi=150, bbox_inches="tight")
    return fig


def select_case_study_days(test_df: pd.DataFrame) -> list[tuple[int, str]]:

    """
    Wybiera 3 reprezentatywne dni ze zbioru testowego do analizy lokalnej:
    1) Dzień załamania pogody,
    2) Dzień bardzo dobrej, ciepłej pogody,
    3) Dzień świąteczny.
    """

    bad_candidates = test_df[test_df["weather"] == "Bad"].sort_values("lag_1", ascending=False)
    bad_idx = int(bad_candidates.index[0])

    clear_candidates = test_df[(test_df["weather"] == "Clear") & (test_df["temp_c"] >= 20)].sort_values(cfg.TARGET, ascending=False)
    clear_idx = int(clear_candidates.index[0])

    holiday_candidates = test_df[test_df["is_holiday"] == 1].sort_values(cfg.TARGET, ascending=True)
    hol_idx = int(holiday_candidates.index[0])

    return [
        (bad_idx, "04a_waterfall_bad_weather.png", "Załamanie pogody (deszcz / burza)"),
        (clear_idx, "04b_waterfall_clear_warm.png", "Ciepły, słoneczny dzień"),
        (hol_idx, "04c_waterfall_holiday.png", "Dzień świąteczny (is_holiday = 1)"),
    ]


def plot_shap_waterfalls(explanation, test_df: pd.DataFrame, save: bool = True) -> list:

    """Wykresy 4a-4c: Lokalna interpretacja SHAP Waterfall dla 3 wybranych dni."""

    ensure_dirs()
    cases = select_case_study_days(test_df)
    figs = []

    for idx, filename, desc in cases:
        row = test_df.loc[idx]
        date_str = pd.to_datetime(row[cfg.DATE]).strftime("%Y-%m-%d")
        actual = int(row[cfg.TARGET])
        pred = float(explanation.base_values[idx] + explanation.values[idx].sum())

        # Zaokrąglenie wartości cech wyświetlanych na osi Y (np. roll_mean_7 do 1 miejsca po przecinku)
        exp_single = explanation[idx]
        formatted_data = []
        for val in exp_single.data:
            if isinstance(val, (float, np.floating)):
                formatted_data.append(round(float(val), 1))
            else:
                formatted_data.append(val)
        exp_clean = shap.Explanation(
            values=exp_single.values,
            base_values=exp_single.base_values,
            data=np.array(formatted_data, dtype=object),
            feature_names=exp_single.feature_names,
        )

        plt.figure(figsize=(11.5, 6.2))
        shap.plots.waterfall(exp_clean, max_display=10, show=False)
        fig = plt.gcf()

        # shap.plots.waterfall tworzy 3 współdzielone osie (główną + dolną E[f(X)] + górną f(x)).
        # Poszerzamy zakres X na wszystkich trzech osiach jednocześnie, aby etykiety ujemnych
        # słupków nie nachodziły na nazwy cech na osi Y.

        xmin, xmax = fig.axes[0].get_xlim()
        span = xmax - xmin
        for a in fig.axes:
            a.set_xlim(xmin - 0.22 * span, xmax + 0.08 * span)

        fig.axes[0].set_title(
            f"{desc} ({date_str})\n"
            f"Rzeczywisty popyt: {actual} | Prognoza modelu: {pred:.0f} | Wczoraj (lag_1): {int(row['lag_1'])}",
            fontsize=11.5, fontweight="bold", pad=28
        )
        fig.tight_layout()

        if save:
            fig.savefig(Path(cfg.SHAP_PLOTS_DIR) / filename, dpi=150, bbox_inches="tight")
        figs.append(fig)

    return figs


def run_all_shap(save: bool = True):

    """Uruchamia pełną analizę SHAP (Etap 3), zapisuje tabele i wykresy."""
    
    explanation, explainer, model, X_test, test_df = compute_shap_values(variant="C")
    impact_df, group_summary = build_shap_impact_table(explanation, X_test, save=save)

    print("=== ETAP 3: PODZIAŁ SIŁY WPŁYWU SHAP (HISTORIA vs ZMIENNE ZEWNĘTRZNE) ===")
    print(group_summary.to_string(index=False))

    print("\n=== RANKING WSZYSTKICH 20 CECH WG ŚREDNIEGO WPŁYWU |SHAP| NA ZBIORZE TESTOWYM ===")
    print(impact_df.to_string(index=False))

    fig1 = plot_shap_beeswarm(explanation, X_test, save=save)
    fig2 = plot_shap_importance_bar(impact_df, group_summary, save=save)
    fig3 = plot_shap_weather_dependence(explanation, X_test, test_df, save=save)
    waterfall_figs = plot_shap_waterfalls(explanation, test_df, save=save)

    if save:
        print(f"\nZapisano tabele SHAP (.parquet) w: {Path(cfg.RESULTS_DIR).resolve()}")
        print(f"Zapisano wykresy SHAP (.png) w: {Path(cfg.SHAP_PLOTS_DIR).resolve()}")

    return explanation, impact_df, group_summary, (fig1, fig2, fig3, *waterfall_figs)


if __name__ == "__main__":
    run_all_shap(save=True)
    plt.close("all")

