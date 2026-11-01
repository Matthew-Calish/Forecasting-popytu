"""
eda.py: Eksploracja danych (EDA) na zbiorze features.parquet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import project_config as cfg
import common_fun as cf

sns.set_theme(style="whitegrid", palette="deep")


def ensure_dirs():
    Path(cfg.EDA_PLOTS_DIR).mkdir(parents=True, exist_ok=True)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Zwraca słownik z tabelami podsumowującymi zbiór danych oraz podział czasowy."""
    train, val, test = cf.split_by_date(df)

    split_summary = pd.DataFrame([
        {
            "Zbiór": name,
            "Od": part[cfg.DATE].min().date(),
            "Do": part[cfg.DATE].max().date(),
            "Liczba dni": len(part),
            "Średni popyt": round(part[cfg.TARGET].mean(), 1),
            "Odch. std popytu": round(part[cfg.TARGET].std(), 1),
            "Min popyt": int(part[cfg.TARGET].min()),
            "Max popyt": int(part[cfg.TARGET].max()),
        }
        for name, part in zip(["train", "val", "test"], [train, val, test])
    ])

    num_cols = [cfg.TARGET, "temp_c", "atemp_c", "humidity", "wind", "lag_1", "lag_7", "roll_mean_7"]
    stats_summary = df[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
    stats_summary["missing"] = df[num_cols].isna().sum()

    return {
        "split_summary": split_summary,
        "stats_summary": stats_summary,
    }


def plot_demand_split(df: pd.DataFrame, save: bool = True):
    """Wykres 1: Szereg czasowy popytu z podziałem na train / val / test."""
    ensure_dirs()
    train, val, test = cf.split_by_date(df)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train[cfg.DATE], train[cfg.TARGET], label="Train (uczacy)", color="#1f77b4", alpha=0.85, lw=1.3)
    ax.plot(val[cfg.DATE], val[cfg.TARGET], label="Val (walidacyjny)", color="#ff7f0e", alpha=0.9, lw=1.5)
    ax.plot(test[cfg.DATE], test[cfg.TARGET], label="Test (testowy)", color="#2ca02c", alpha=0.9, lw=1.5)

    # 14-dniowa średnia krocząca dla czytelności trendu
    ax.plot(
        df[cfg.DATE],
        df["roll_mean_14"],
        label="Średnia krocząca 14-dniowa (roll_mean_14)",
        color="#d62728",
        linestyle="--",
        lw=1.8,
    )

    ax.axvline(pd.to_datetime(cfg.TRAIN_END), color="black", linestyle=":", lw=1.2, label=f"Koniec Train ({cfg.TRAIN_END})")
    ax.axvline(pd.to_datetime(cfg.VAL_END), color="gray", linestyle=":", lw=1.2, label=f"Koniec Val ({cfg.VAL_END})")

    ax.set_title("Dzienny popyt (demand) w latach 2011–2012 i podział czasowy danych", fontsize=13, fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel("Dzienny popyt (liczba wypożyczeń)")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.EDA_PLOTS_DIR) / "01_demand_split.png", dpi=150)
    return fig


def plot_weather_season_boxplots(df: pd.DataFrame, save: bool = True):
    """Wykres 2: Rozkład popytu względem pory roku, pogody oraz typu dnia."""
    ensure_dirs()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    sns.boxplot(
        data=df, x="season", y=cfg.TARGET, ax=axes[0],
        order=cfg.SEASONS, palette="Set2", hue="season", legend=False
    )
    axes[0].set_title("Popyt a pora roku (season)", fontweight="bold")
    axes[0].set_xlabel("Pora roku")
    axes[0].set_ylabel("Dzienny popyt")

    sns.boxplot(
        data=df, x="weather", y=cfg.TARGET, ax=axes[1],
        order=cfg.WHEATER, palette="Blues_r", hue="weather", legend=False
    )
    axes[1].set_title("Popyt a pogoda (weather)", fontweight="bold")
    axes[1].set_xlabel("Warunki pogodowe")
    axes[1].set_ylabel("")

    day_type = np.where(
        df["is_holiday"] == 1, "Święto",
        np.where(df["is_weekend"] == 1, "Weekend", "Dzień roboczy")
    )
    temp_df = df.assign(day_type=day_type)
    sns.boxplot(
        data=temp_df, x="day_type", y=cfg.TARGET, ax=axes[2],
        order=["Dzień roboczy", "Weekend", "Święto"], palette="Pastel1", hue="day_type", legend=False
    )
    axes[2].set_title("Popyt a typ dnia (kalendarz)", fontweight="bold")
    axes[2].set_xlabel("Typ dnia")
    axes[2].set_ylabel("")

    fig.suptitle("Wpływ zmiennych zewnętrznych (sezon, pogoda, kalendarz) na rozkład popytu", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.EDA_PLOTS_DIR) / "02_weather_season_boxplots.png", dpi=150, bbox_inches="tight")
    return fig


def plot_temp_humidity_scatter(df: pd.DataFrame, save: bool = True):
    """Wykres 3: Zależność popytu od temperatury i wilgotności z podziałem na pogodę."""
    ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)

    sns.scatterplot(
        data=df, x="temp_c", y=cfg.TARGET, hue="weather",
        hue_order=cfg.WHEATER, alpha=0.75, s=45, ax=axes[0]
    )
    sns.regplot(
        data=df, x="temp_c", y=cfg.TARGET, scatter=False, order=2,
        color="black", ax=axes[0], line_kws={"linestyle": "--", "label": "Trend nieliniowy (st. 2)"}
    )
    axes[0].set_title("Popyt w funkcji temperatury (temp_c)", fontweight="bold")
    axes[0].set_xlabel("Temperatura [°C]")
    axes[0].set_ylabel("Dzienny popyt")

    sns.scatterplot(
        data=df, x="humidity", y=cfg.TARGET, hue="weather",
        hue_order=cfg.WHEATER, alpha=0.75, s=45, ax=axes[1]
    )
    sns.regplot(
        data=df, x="humidity", y=cfg.TARGET, scatter=False,
        color="black", ax=axes[1], line_kws={"linestyle": "--"}
    )
    axes[1].set_title("Popyt w funkcji wilgotności (humidity)", fontweight="bold")
    axes[1].set_xlabel("Wilgotność względna [%]")
    axes[1].set_ylabel("")

    fig.suptitle("Nieliniowy wpływ temperatury oraz negatywny wpływ wysokiej wilgotności i opadów", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.EDA_PLOTS_DIR) / "03_temp_humidity_scatter.png", dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, save: bool = True):
    """Wykres 4: Macierz korelacji cech numerycznych z popytem."""
    ensure_dirs()
    cols = [
        cfg.TARGET, "lag_1", "lag_7", "lag_14",
        "roll_mean_7", "roll_mean_14", "roll_std_7",
        "temp_c", "atemp_c", "temp_lag_1", "humidity", "wind", "rain_flag",
    ]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Macierz korelacji cech historycznych i pogodowych ze zmienną celu (demand)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig(Path(cfg.EDA_PLOTS_DIR) / "04_correlation_matrix.png", dpi=150)
    return fig


def run_all_eda(save: bool = True):
    """Uruchamia pełne EDA, zapisuje wykresy i wypisuje statystyki."""
    df = cf.load_model_data()
    summary = get_dataset_summary(df)

    print("=== PODZIAŁ CZASOWY DANYCH (features.parquet) ===")
    print(summary["split_summary"].to_string(index=False))
    print("\n=== STATYSTYKI KLUCZOWYCH ZMIENNYCH ===")
    print(summary["stats_summary"].to_string())

    fig1 = plot_demand_split(df, save=save)
    fig2 = plot_weather_season_boxplots(df, save=save)
    fig3 = plot_temp_humidity_scatter(df, save=save)
    fig4 = plot_correlation_matrix(df, save=save)

    if save:
        print(f"\nZapisano 4 wykresy EDA w katalogu: {Path(cfg.EDA_PLOTS_DIR).resolve()}")

    return df, summary, (fig1, fig2, fig3, fig4)


if __name__ == "__main__":
    run_all_eda(save=True)
    plt.close("all")

