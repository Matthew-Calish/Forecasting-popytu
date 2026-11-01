"""
WSPÓLNA KONFIGURACJA PROJEKTU

Proszę o importowanie danych z tego pliku bo to Single Source of Truth dla całego projektu.

Jeśli ktoś zmienia coś TUTAJ, zmiana obowiązuje cały projekt.

"""

# WAŻNE PLIKI I KATALOGI
RAW_CSV = "data_raw.csv"
DATA_CSV = "data_clean.csv"
MODEL_DATA = "features.parquet"

EDA_PLOTS_DIR = "plots/eda"
MODEL_PLOTS_DIR = "plots/models"
SHAP_PLOTS_DIR = "plots/shap"
RESULTS_DIR = "results"


# WAŻNE POLA
DATE = "date"
TARGET = "demand"


# LAGI I OKNA
LAGS = [1, 7, 14]                     # popyt sprzed 1, 7 i 14 dni
ROLL_MEAN_WINDOWS = [7, 14]           # średnia z ostatnich 7 i 14 dni
ROLL_STD_WINDOWS = [7]                # odchylenie z ostatnich 7 dni


# ODCIĘTE DNI (bo nie mają pełnych danych historycznych - lagów i okien)
WARMUP_DAYS = max(LAGS + ROLL_MEAN_WINDOWS + ROLL_STD_WINDOWS)   # = 14


# KATEGORIE
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
WHEATER = ["Clear", "Cloudy", "Bad"]
CATEGORIES = ["season", "weather"]


# PODZIAŁ CZASOWY DANYCH
TRAIN_END = "2012-05-31" # koniec treningu modelu
VAL_END = "2012-08-31" # koniec walidacji modelu (początek testów)


# WARIANTY / ABLACJA

# A: tylko kalendarz + historia popytu (model nie wie nic o świecie zewnętrznym)
A = [
    "weekday", "is_weekend", "month",
    "lag_1", "lag_7", "lag_14",
    "roll_mean_7", "roll_std_7", "roll_mean_14",
]

# B: A + zmienne zewnętrzne (pogoda i święta).
B = A + [
    "temp_c", "atemp_c", "humidity", "wind", "weather",
    "is_holiday", "is_workday",
]

# C: B + cechy pochodne z pogody
C = B + [
    "temp_lag_1", "rain_flag", "temp_x_weekend", "season",
]

VARIANTS = {"A": A, "B": B, "C": C}
EXTERNAL_COLS = [c for c in C if c not in A]


# MODELE BAZOWE (BASELINE)
BASELINES = {
    "Baseline_Lag1": "lag_1",
    "Baseline_Lag7": "lag_7",
    "Baseline_RollMean7": "roll_mean_7",
}

