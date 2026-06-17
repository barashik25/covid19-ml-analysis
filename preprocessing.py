import pandas as pd
import numpy as np

print("=" * 60)
print("ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)


# Dataset 1: JRC — статистика COVID-19 по странам
print("\n--- Загрузка датасета 1 (JRC) ---")
df1 = pd.read_csv("jrc-covid-19-all-days-by-country.csv")
print(f"Исходный размер: {df1.shape[0]} строк, {df1.shape[1]} столбцов")

# Удаляем дубликаты
df1.drop_duplicates(inplace=True)

# Оставляем только нужные столбцы
cols1 = [
    "Date", "iso3", "CountryName",
    "CumulativePositive", "CumulativeDeceased",
    "CumulativeRecovered", "CurrentlyPositive",
    "Hospitalized", "IntensiveCare"
]
df1 = df1[cols1]

# Оставляем только строки где есть данные о госпитализации
df1 = df1[df1["Hospitalized"].notna()].copy()
print(f"После фильтрации (есть данные о госпитализации): {df1.shape[0]} строк")

# Заполняем пропуски числовых признаков медианой
for col in ["CumulativePositive", "CumulativeDeceased",
            "CumulativeRecovered", "CurrentlyPositive"]:
    median_val = df1[col].median()
    df1[col] = df1[col].fillna(median_val)

# IntensiveCare заполняем отдельно: сначала сортируем по стране и дате,
# чтобы строки шли хронологически, затем для каждой страны заполняем пропуски
# значением предыдущего дня (ffill), а если в начале тоже пусто — следующего (bfill).
# Это точнее медианы, так как данные реанимации меняются плавно день за днём.
df1 = df1.sort_values(["iso3", "Date"])
df1["IntensiveCare"] = (
    df1.groupby("iso3")["IntensiveCare"]
    .transform(lambda x: x.ffill().bfill())
)
# Если осталось — заполняем медианой
df1["IntensiveCare"] = df1["IntensiveCare"].fillna(df1["IntensiveCare"].median())

# Создаём целевую переменную:
# high_hospitalization = 1, если госпитализированных > медианы
threshold = df1["Hospitalized"].median()
df1["high_hospitalization"] = (df1["Hospitalized"] > threshold).astype(int)
print(f"\nПорог для высокой госпитализации: {threshold:.0f} чел.")  #показать просто целое число
print("Распределение целевой переменной (датасет 1):")
print(df1["high_hospitalization"].value_counts())

# Убираем служебные столбцы перед обучением
df1_model = df1.drop(columns=["Date", "iso3", "CountryName", "Hospitalized"])

# Сохраняем
df1_model.to_csv("df1_processed.csv", index=False)
print(f"\nДатасет 1 сохранён: df1_processed.csv ({df1_model.shape[0]} строк)")





# ДАТАСЕТ 2: GDSI — пациенты с рассеянным склерозом
print("\n--- Загрузка датасета 2 (GDSI / MS) ---")
df2 = pd.read_csv("GDSI_OpenDataset_Final.csv")
print(f"Исходный размер: {df2.shape[0]} строк, {df2.shape[1]} столбцов")

# Удаляем дубликаты и служебные столбцы
df2.drop_duplicates(inplace=True)
df2.drop(columns=["secret_name", "stop_or_end_date_combined",
                   "comorbidities_other", "has_comorbidities.1",
                   "has_comorbidities.2"], inplace=True, errors="ignore")

# --- Целевая переменная ---
# Удаляем строки где целевая переменная отсутствует
df2 = df2[df2["covid19_admission_hospital"].notna()].copy()
df2["target"] = (df2["covid19_admission_hospital"] == "yes").astype(int)
df2.drop(columns=["covid19_admission_hospital"], inplace=True)

print("\nРаспределение целевой переменной (датасет 2):")
print(df2["target"].value_counts())
print(f"Дисбаланс классов: {df2['target'].mean()*100:.1f}% положительных случаев")

# --- Обработка пропусков ---

# Бинарные признаки Yes/No → 1/0 (пропуск = отсутствие)
binary_cols = [
    "covid19_sympt_chills", "covid19_sympt_dry_cough", "covid19_sympt_fatigue",
    "covid19_sympt_fever", "covid19_sympt_loss_smell_taste",
    "covid19_sympt_nasal_congestion", "covid19_sympt_pain",
    "covid19_sympt_pneumonia", "covid19_sympt_shortness_breath",
    "covid19_sympt_sore_throat", "covid19_ventilation",
    "com_cardiovascular_disease", "com_diabetes", "com_hypertension",
    "com_immunodeficiency", "com_lung_disease", "com_malignancy",
    "com_neurological_neuromuscular", "com_chronic_kidney_disease",
    "com_chronic_liver_disease", "covid19_icu_stay",
    "covid19_has_symptoms", "covid19_self_isolation",
    "covid19_outcome_recovered", "current_dmt",
    "dmt_glucocorticoid", "current_or_former_smoker", "has_comorbidities"
]
for col in binary_cols:
    if col in df2.columns:
        df2[col] = df2[col].fillna("No")
        df2[col] = df2[col].map(
            lambda x: 1 if str(x).strip().lower() in ["yes", "1"] else 0
        )

# Категориальные признаки → мода
mode_cols = ["bmi_in_cat2", "dmt_type_overall", "ms_type2",
             "edss_in_cat2", "duration_treatment_cat2",
             "covid19_diagnosis", "covid19_confirmed_case",
             "report_source"]
for col in mode_cols:
    if col in df2.columns:
        mode_val = df2[col].mode()
        if len(mode_val) > 0:
            df2[col] = df2[col].fillna(mode_val[0])

# Числовой признак year_onset → медиана
if "year_onset" in df2.columns:
    df2["year_onset"] = df2["year_onset"].fillna(df2["year_onset"].median())

# Признак pregnancy — отдельная категория
if "pregnancy" in df2.columns:
    df2["pregnancy"] = df2["pregnancy"].fillna("unknown")

# Кодируем оставшиеся категориальные признаки (One-Hot Encoding)
cat_cols = df2.select_dtypes(include="object").columns.tolist()
if "target" in cat_cols:
    cat_cols.remove("target")

print(f"\nКатегориальные признаки для кодирования: {cat_cols}")
df2 = pd.get_dummies(df2, columns=cat_cols, drop_first=False)

# Итоговая проверка
print(f"\nПропуски после обработки: {df2.isnull().sum().sum()}")
print(f"Итоговый размер датасета 2: {df2.shape[0]} строк, {df2.shape[1]} столбцов")

# Сохраняем
df2.to_csv("df2_processed.csv", index=False)
print(f"Датасет 2 сохранён: df2_processed.csv")

print("\n" + "=" * 60)
print("Предобработка завершена!")
print("Следующий шаг: запусти model_training.py")
print("=" * 60)