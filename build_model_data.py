"""
build_model_data.py: z wyczyszczonego CSV robi dane do modelu.
    
"""
import numpy as np
import pandas as pd
import common_fun as cf
import project_config as cfg


# DODANIE CECH HISTORYCZNYCH (lagi i okna) + cechy pochodne z pogody

def add_features(df):

    df = df.copy()
    target = df[cfg.TARGET]


    # LAGI
    for lag in cfg.LAGS:
        df[f"lag_{lag}"] = target.shift(lag)


    # OKNA (średnia i odchylenie z ostatnich dni)
    past = target.shift(1)

    for win in cfg.ROLL_MEAN_WINDOWS:
        df[f"roll_mean_{win}"] = past.rolling(win).mean()

    for win in cfg.ROLL_STD_WINDOWS:
        df[f"roll_std_{win}"] = past.rolling(win).std()


    # CECHY POCHODNE Z POGODY DLA WARIANTU C
    df["temp_lag_1"] = df["temp_c"].shift(1)                    # temperatura wczoraj
    df["rain_flag"] = (df["weather"] == "Bad").astype(int)      # opady / burza
    df["temp_x_weekend"] = df["temp_c"] * df["is_weekend"]      # interakcja (idk)


    # obcięcie początku danych, bo nie ma pełnych lagów i okien
    df = df.iloc[cfg.WARMUP_DAYS:].reset_index(drop=True)

    return df


# TEST POPRAWNOŚCI CECH HISTORYCZNYCH (brak wycieku danych z przyszłości)

def check_no_leakage(data, feat):

    target = data.set_index(cfg.DATE)[cfg.TARGET]
    dates = feat[cfg.DATE]

    for lag in cfg.LAGS:
        expected = target.reindex(dates - pd.Timedelta(days=lag)).to_numpy()
        assert np.allclose(feat[f"lag_{lag}"].to_numpy(), expected), f"lag_{lag} niezgodny"

    for win in cfg.ROLL_MEAN_WINDOWS:
        # okno = dni od (data - w) do (data - 1) włącznie; dzisiejszego dnia tam NIE MA
        expected = [target.loc[d - pd.Timedelta(days=win): d - pd.Timedelta(days=1)].mean() for d in dates]
        assert np.allclose(feat[f"roll_mean_{win}"].to_numpy(), expected), f"roll_mean_{win} niezgodny"

    for win in cfg.ROLL_STD_WINDOWS:
        expected = [target.loc[d - pd.Timedelta(days=win): d - pd.Timedelta(days=1)].std() for d in dates]
        assert np.allclose(feat[f"roll_std_{win}"].to_numpy(), expected), f"roll_std_{win} niezgodny"

    # cechy historyczne nie mogą mieć braków po obcięciu początku
    hist_cols = [c for c in feat.columns if c.startswith(("lag_", "roll_"))] + ["temp_lag_1"]

    assert feat[hist_cols].notna().all().all(), "Braki w cechach historycznych"



# MAIN
def main():

    # WCZYTANIE i SPRAWDZENIE
    data = cf.load_clean()
    cf.check_input(data)

    # DODANIE CECH i SPRAWDZENIE BRAKU WYCIEKU
    features = add_features(data)
    check_no_leakage(data, features)

    # ZAPIS DO PARQUET
    features.to_parquet(cfg.MODEL_DATA, index=False)


    print(f"Zapisano {cfg.MODEL_DATA}: {features.shape[0]} wierszy, {features.shape[1]} kolumn")

    print()

    # WYŚWIETLENIE OKRESÓW
    for name, part in zip(["train", "val", "test"], cf.split_by_date(features)):
        print(f"{name:5s} {part[cfg.DATE].min().date()} -> {part[cfg.DATE].max().date()}  ({len(part)} dni)")
    print()

    # WYŚWIETLENIE WARIANTÓW
    for name, cols in cfg.VARIANTS.items():
        print(f"Wariant {name}: {len(cols)} cech")
    print("\nWszystkie testy (daty, wyciek danych, warianty) przeszły.")


if __name__ == "__main__":
    main()
