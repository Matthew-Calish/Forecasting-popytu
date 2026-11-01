# Forecasting popytu z feature engineering + SHAP

Prognozowanie dziennego popytu na podstawie danych historycznych oraz zmiennych zewnętrznych (pogoda, kalendarz, święta) wraz z analizą ablacyjną i interpretowalnością **SHAP**.

---

## Cel projektu
Zbudowanie praktycznego modelu **LightGBM** do prognozy dziennego popytu oraz ilościowe zbadanie, jak zmienne zewnętrzne (temperatura, wilgotność, wiatr, opady, święta) wpływają na dokładność prognozy względem modelu opartego wyłącznie na historii popytu.

---

## Struktura projektu

- `project_config.py` – centralna konfiguracja projektu (*Single Source of Truth*): ścieżki, podział czasowy, definicje wariantów cech (`A`, `B`, `C`) oraz hiperparametry modelu.
- `common_fun.py` – wspólne funkcje pomocnicze (wczytywanie danych, walidacja spójności dat, podział `train` / `val` / `test`).
- `data_cleaning.py` – czyszczenie surowych danych (`data_raw.csv` $\rightarrow$ `data_clean.csv`) i denormalizacja jednostek fizycznych (°C, %, km/h).
- `build_model_data.py` – *Feature Engineering* (`data_clean.csv` $\rightarrow$ `features.parquet`): wyliczanie opóźnień (`lag_1`, `lag_7`, `lag_14`), okien kroczących (`roll_mean_7`, `roll_mean_14`, `roll_std_7`), cech pogodowych oraz test braku wycieku danych (`check_no_leakage`).
- `eda.py` – eksploracja danych i generowanie wykresów do `plots/eda/`.
- `train_models.py` – ewaluacja modeli bazowych (*Baseline*) oraz trening `LightGBM` w wariantach **A**, **B**, **C** (zapis wyników w `results/` i wykresów w `plots/models/`).
- `shap_analysis.py` – analiza interpretowalności `shap.TreeExplainer` (zapis tabel w `results/` i wykresów w `plots/shap/`).
- `demo_analysis.ipynb` – zbiorczy notebook prezentacyjny łączący opis, tabele wyników oraz wszystkie wykresy EDA, ablacji i SHAP.

---

## Warianty cech (Badanie ablacyjne)

| Wariant | Liczba cech | Opis zestawu cech | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `Baseline_Lag1` | 1 | Prognoza naiwna: popyt z wczoraj (`lag_1`) | 916.40 | 1330.39 | 0.5391 |
| **Wariant A** | 9 | Tylko kalendarz + historia popytu (lagi i średnie kroczące) | 999.57 | 1347.56 | 0.5271 |
| **Wariant B** | 16 | Wariant A + pogoda i święta (`temp_c`, `atemp_c`, `humidity`, `wind`, `weather`, `is_holiday`, `is_workday`) | 835.25 | 1064.24 | 0.7051 |
| **Wariant C** | 20 | Wariant B + cechy pochodne pogody (`temp_lag_1`, `rain_flag`, `temp_x_weekend`, `season`) | **804.29** | **1033.48** | **0.7219** |

**Główne wnioski:**
- Dodanie zmiennych zewnętrznych i cech pochodnych (**Wariant C**) zmniejsza błąd **MAE o 19.54%** względem modelu bez pogody (**Wariant A**) oraz o **12.23%** względem najlepszego modelu bazowego (`Baseline_Lag1`).
- W dni deszczowe (`weather = Bad`) błąd MAE spada o **50%** (z `3097` w Wariancie A do `1552` w Wariancie C).
- Analiza **SHAP** wykazuje, że cechy historyczne odpowiadają za **59.7%** siły decyzyjnej modelu, a zmienne zewnętrzne za **40.3%** (z czego najważniejsze to `humidity`, `temp_c`, `weather`, `atemp_c` oraz `wind`).

---

## Jak uruchomić projekt

0. Utworzenie i aktywacja środowiska wirtualnego:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
1. Instalacja zależności w środowisku wirtualnym:
   ```bash
   pip install -r requirements.txt
   ```
2. Przygotowanie danych i cech (opcjonalnie – pliki `data_clean.csv` i `features.parquet` są już w repozytorium):
   ```bash
   python data_cleaning.py
   python build_model_data.py
   ```
3. Wygenerowanie wykresów EDA, trening modeli i analiza SHAP z konsoli:
   ```bash
   python eda.py
   python train_models.py
   python shap_analysis.py
   ```
4. Prezentacja wyników: otwórz notebook `demo_analysis.ipynb`.
