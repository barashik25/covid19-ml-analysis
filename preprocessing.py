import pandas as pd
import numpy as np

# --- Dataset 1: JRC country-level COVID-19 statistics ---
print("Loading dataset 1 (JRC)...")
df1 = pd.read_csv("jrc-covid-19-all-days-by-country.csv")
print(f"Original shape: {df1.shape[0]} rows, {df1.shape[1]} columns")

df1.drop_duplicates(inplace=True)

# keep only the columns we actually need
cols1 = [
    "Date", "iso3", "CountryName",
    "CumulativePositive", "CumulativeDeceased",
    "CumulativeRecovered", "CurrentlyPositive",
    "Hospitalized", "IntensiveCare"
]
df1 = df1[cols1]

# only rows that actually have hospitalization data are useful for the target
df1 = df1[df1["Hospitalized"].notna()].copy()
print(f"After dropping rows without hospitalization data: {df1.shape[0]} rows")

# fill missing values in the numeric columns with the median
for col in ["CumulativePositive", "CumulativeDeceased",
            "CumulativeRecovered", "CurrentlyPositive"]:
    df1[col] = df1[col].fillna(df1[col].median())

# IntensiveCare gets a different treatment: sort by country and date so rows
# are chronological, then forward/backward fill within each country. ICU
# numbers change gradually day to day, so this is more realistic than just
# plugging in the median everywhere.
df1 = df1.sort_values(["iso3", "Date"])
df1["IntensiveCare"] = (
    df1.groupby("iso3")["IntensiveCare"]
    .transform(lambda x: x.ffill().bfill())
)
# anything still missing (e.g. a country with no ICU data at all) -> median
df1["IntensiveCare"] = df1["IntensiveCare"].fillna(df1["IntensiveCare"].median())

# target variable: 1 if hospitalized count is above the median, else 0
threshold = df1["Hospitalized"].median()
df1["high_hospitalization"] = (df1["Hospitalized"] > threshold).astype(int)
print(f"Threshold for 'high hospitalization': {threshold:.0f}")
print(df1["high_hospitalization"].value_counts())

# drop identifier/date columns before feeding this into a model
df1_model = df1.drop(columns=["Date", "iso3", "CountryName", "Hospitalized"])

df1_model.to_csv("df1_processed.csv", index=False)
print(f"Saved df1_processed.csv ({df1_model.shape[0]} rows)\n")


# --- Dataset 2: GDSI, multiple sclerosis patients ---
print("Loading dataset 2 (GDSI / MS)...")
df2 = pd.read_csv("GDSI_OpenDataset_Final.csv")
print(f"Original shape: {df2.shape[0]} rows, {df2.shape[1]} columns")

df2.drop_duplicates(inplace=True)
df2.drop(columns=["secret_name", "stop_or_end_date_combined",
                   "comorbidities_other", "has_comorbidities.1",
                   "has_comorbidities.2"], inplace=True, errors="ignore")

# target variable
df2 = df2[df2["covid19_admission_hospital"].notna()].copy()
df2["target"] = (df2["covid19_admission_hospital"] == "yes").astype(int)
df2.drop(columns=["covid19_admission_hospital"], inplace=True)

print(df2["target"].value_counts())
print(f"Class imbalance: {df2['target'].mean()*100:.1f}% positive cases")

# --- missing values ---

# yes/no columns -> 1/0, missing counted as "no"
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

# categorical columns -> mode
mode_cols = ["bmi_in_cat2", "dmt_type_overall", "ms_type2",
             "edss_in_cat2", "duration_treatment_cat2",
             "covid19_diagnosis", "covid19_confirmed_case",
             "report_source"]
for col in mode_cols:
    if col in df2.columns:
        mode_val = df2[col].mode()
        if len(mode_val) > 0:
            df2[col] = df2[col].fillna(mode_val[0])

# year_onset is numeric -> median
if "year_onset" in df2.columns:
    df2["year_onset"] = df2["year_onset"].fillna(df2["year_onset"].median())

# pregnancy gets its own "unknown" bucket instead of a guess
if "pregnancy" in df2.columns:
    df2["pregnancy"] = df2["pregnancy"].fillna("unknown")

# one-hot encode whatever categorical columns are left
cat_cols = df2.select_dtypes(include="object").columns.tolist()
if "target" in cat_cols:
    cat_cols.remove("target")

print(f"Encoding categorical columns: {cat_cols}")
df2 = pd.get_dummies(df2, columns=cat_cols, drop_first=False)

print(f"Missing values remaining: {df2.isnull().sum().sum()}")
print(f"Final shape of dataset 2: {df2.shape[0]} rows, {df2.shape[1]} columns")

df2.to_csv("df2_processed.csv", index=False)
print("Saved df2_processed.csv")

print("\nPreprocessing done, next step is model_training.py")
