import pandas as pd, numpy as np
import project_config as cfg

data = pd.read_csv(cfg.RAW_CSV, parse_dates=["dteday"])
data = data.sort_values("dteday").reset_index(drop=True)

# ZMIANA NAZW KOLUMN
data = data.rename(columns={"dteday": "date", "cnt": "demand", "mnth": "month"})

# ZMIANA SEZONÓW
season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
data["season"]  = data["season"].map(season_map)
data["season"]  = data["season"].astype("category")

# ZMIANA POGODY
weather_map = {1: "Clear", 2: "Cloudy", 3: "Bad", 4: "Very_bad"}
data["weather"] = data["weathersit"].map(weather_map)
data["weather"] = data["weather"].astype("category")

# DENORMALIZACJA JEDNOSTEK
data["temp_c"]    = (data["temp"]  * 47 - 8).round(1)     # °C
data["atemp_c"]   = (data["atemp"] * 66 - 16).round(1)     # °C (odczuwalna)
data["humidity"] = (data["hum"].replace(0, np.nan) * 100).round(1)
data["wind"]      = (data["windspeed"] * 67).round(1)  # 

# ZMIANA DNI TYGODNIA
data["weekday"]  = data["date"].dt.day_of_week + 1

# ZMIANA FLAG NA BOOLEAN
data["is_weekend"] = (data["weekday"] >= 6).astype(int)
data["is_workday"] = (data["workingday"] == 1).astype(int)
data["is_holiday"] = (data["holiday"] == 1).astype(int)

# USUNIĘCIE NIEPOTRZEBNYCH KOLUMN
data = data.drop(columns=["instant", "weathersit",
                           "workingday", "holiday", "temp", 
                           "atemp", "casual", "registered", 
                           "yr", "hum", "windspeed"
                           ])

# OKREŚLENIE KOLEJNOŚCI KOLUMN
data = data[["date", "month", "weekday", 
             "is_workday", "is_weekend", "is_holiday",
               "season", "weather", "temp_c", 
             "atemp_c", "humidity", "wind", "demand"
             ]]

# ZAPIS DO PLIKU
data.to_csv(cfg.DATA_CSV, index=False, na_rep="NaN")