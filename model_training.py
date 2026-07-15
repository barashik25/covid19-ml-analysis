import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE


def print_metrics(name, y_true, y_pred, y_prob):
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_true, y_prob):.4f}")


def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  saved {filename}")


# ------------------------------------------------------------------
# Dataset 1: JRC country-level data
# ------------------------------------------------------------------
print("Dataset 1: JRC (country-level)")

df1 = pd.read_csv("df1_processed.csv")
X1 = df1.drop(columns=["high_hospitalization"])
y1 = df1["high_hospitalization"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1
)
print(f"Train: {X1_train.shape[0]} rows, test: {X1_test.shape[0]} rows")

# Random Forest
rf1 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf1.fit(X1_train, y1_train)
y1_pred_rf = rf1.predict(X1_test)
y1_prob_rf = rf1.predict_proba(X1_test)[:, 1]
print_metrics("Random Forest", y1_test, y1_pred_rf, y1_prob_rf)
save_confusion_matrix(y1_test, y1_pred_rf,
    "Random Forest - Dataset 1", "cm_df1_rf.png")

# MLP
mlp1 = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
)
mlp1.fit(X1_train, y1_train)
y1_pred_mlp = mlp1.predict(X1_test)
y1_prob_mlp = mlp1.predict_proba(X1_test)[:, 1]
print_metrics("MLP", y1_test, y1_pred_mlp, y1_prob_mlp)
save_confusion_matrix(y1_test, y1_pred_mlp,
    "MLP - Dataset 1", "cm_df1_mlp.png")

# feature importance (RF)
importances1 = pd.Series(rf1.feature_importances_, index=X1.columns)
top10_1 = importances1.sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
top10_1.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 10 features - Dataset 1 (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_df1.png")
plt.close()
print("saved feature_importance_df1.png")


# ------------------------------------------------------------------
# Dataset 2: GDSI, multiple sclerosis patients
# ------------------------------------------------------------------
print("\nDataset 2: GDSI (MS patients)")

df2 = pd.read_csv("df2_processed.csv")
X2 = df2.drop(columns=["target"])
y2 = df2["target"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)
print(f"Train before SMOTE: {X2_train.shape[0]} rows ({y2_train.sum()} hospitalized)")

# SMOTE only on the training set - the test set has to reflect the real distribution
smote = SMOTE(random_state=42)
X2_train_sm, y2_train_sm = smote.fit_resample(X2_train, y2_train)
print(f"Train after SMOTE: {X2_train_sm.shape[0]} rows ({y2_train_sm.sum()} hospitalized)")
print(f"Test: {X2_test.shape[0]} rows")

# Random Forest
rf2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf2.fit(X2_train_sm, y2_train_sm)
y2_pred_rf = rf2.predict(X2_test)
y2_prob_rf = rf2.predict_proba(X2_test)[:, 1]
print_metrics("Random Forest", y2_test, y2_pred_rf, y2_prob_rf)
save_confusion_matrix(y2_test, y2_pred_rf,
    "Random Forest - Dataset 2", "cm_df2_rf.png")

# MLP
mlp2 = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
)
mlp2.fit(X2_train_sm, y2_train_sm)
y2_pred_mlp = mlp2.predict(X2_test)
y2_prob_mlp = mlp2.predict_proba(X2_test)[:, 1]
print_metrics("MLP", y2_test, y2_pred_mlp, y2_prob_mlp)
save_confusion_matrix(y2_test, y2_pred_mlp,
    "MLP - Dataset 2", "cm_df2_mlp.png")

# feature importance (RF)
importances2 = pd.Series(rf2.feature_importances_, index=X2.columns)
top10_2 = importances2.sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
top10_2.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 10 features - Dataset 2 (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_df2.png")
plt.close()
print("saved feature_importance_df2.png")


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
results = pd.DataFrame({
    "Dataset":  ["JRC", "JRC", "GDSI/MS", "GDSI/MS"],
    "Model":    ["Random Forest", "MLP", "Random Forest", "MLP"],
    "Accuracy": [
        accuracy_score(y1_test, y1_pred_rf),
        accuracy_score(y1_test, y1_pred_mlp),
        accuracy_score(y2_test, y2_pred_rf),
        accuracy_score(y2_test, y2_pred_mlp),
    ],
    "Precision": [
        precision_score(y1_test, y1_pred_rf, zero_division=0),
        precision_score(y1_test, y1_pred_mlp, zero_division=0),
        precision_score(y2_test, y2_pred_rf, zero_division=0),
        precision_score(y2_test, y2_pred_mlp, zero_division=0),
    ],
    "Recall": [
        recall_score(y1_test, y1_pred_rf, zero_division=0),
        recall_score(y1_test, y1_pred_mlp, zero_division=0),
        recall_score(y2_test, y2_pred_rf, zero_division=0),
        recall_score(y2_test, y2_pred_mlp, zero_division=0),
    ],
    "F1": [
        f1_score(y1_test, y1_pred_rf, zero_division=0),
        f1_score(y1_test, y1_pred_mlp, zero_division=0),
        f1_score(y2_test, y2_pred_rf, zero_division=0),
        f1_score(y2_test, y2_pred_mlp, zero_division=0),
    ],
    "AUC-ROC": [
        roc_auc_score(y1_test, y1_prob_rf),
        roc_auc_score(y1_test, y1_prob_mlp),
        roc_auc_score(y2_test, y2_prob_rf),
        roc_auc_score(y2_test, y2_prob_mlp),
    ],
})
results = results.round(4)
print("\n" + results.to_string(index=False))
results.to_csv("results_summary.csv", index=False)
print("\nSaved results_summary.csv")
