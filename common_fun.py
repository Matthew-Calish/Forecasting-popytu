import pandas as pd
import project_config as cfg

'''
    FUNKCJE POMOCNICZE DLA WSZYTSKICH
'''


# Wczytywanie danych z CSV
def load_clean(path=cfg.DATA_CSV):

    df = pd.read_csv(path, parse_dates=[cfg.DATE])
    df = df.sort_values(cfg.DATE).reset_index(drop=True)   # sortowanie = obowiązkowe

    # kategorie ze stałym zestawem wartości
    df["season"] = pd.Categorical(df["season"], categories=cfg.SEASONS)
    df["weather"] = pd.Categorical(df["weather"], categories=cfg.WHEATER)

    return df

# Wczytywanie danych z Parquet
def load_model_data(path=cfg.MODEL_DATA):
    return pd.read_parquet(path)


# Sprawdza poprawnosc csv
def check_input(df):

    full_range = pd.date_range(df[cfg.DATE].min(), df[cfg.DATE].max())
    
    assert len(full_range) == len(df), "Dane mają dziury w datach, shift() liczyłby złe dni!"
    assert df[cfg.DATE].is_unique, "Zduplikowane daty"
    assert df[cfg.TARGET].notna().all(), "Braki w zmiennej celu"
    assert df["season"].notna().all() and df["weather"].notna().all(), \
        "Nieznana wartość w season/weather (sprawdź kategorie w config.py)"


# Podział na okresy
def split_by_date(df):

    """Dzieli dane wg dat z config.py. Zwraca (train, val, test)."""

    d = df[cfg.DATE]
    train = df[d <= cfg.TRAIN_END]
    val = df[(d > cfg.TRAIN_END) & (d <= cfg.VAL_END)]
    test = df[d > cfg.VAL_END]
    return train, val, test


# Na pewno jest to użyteczne
def get_xy(df, variant):

    """Zwraca (X, y) dla wariantu 'A', 'B' lub 'C'."""

    cols = cfg.VARIANTS[variant]
    return df[cols], df[cfg.TARGET]