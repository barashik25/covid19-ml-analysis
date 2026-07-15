import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE

# retrain both models the same way as in model_training.py, just to get
# probability scores for the ROC curves

# Dataset 1
df1 = pd.read_csv("df1_processed.csv")
X1 = df1.drop(columns=["high_hospitalization"])
y1 = df1["high_hospitalization"]
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1)

rf1 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf1.fit(X1_train, y1_train)
y1_prob_rf = rf1.predict_proba(X1_test)[:, 1]

mlp1 = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                     max_iter=500, random_state=42)
mlp1.fit(X1_train, y1_train)
y1_prob_mlp = mlp1.predict_proba(X1_test)[:, 1]

# Dataset 2
df2 = pd.read_csv("df2_processed.csv")
X2 = df2.drop(columns=["target"])
y2 = df2["target"]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2)

smote = SMOTE(random_state=42)
X2_train_sm, y2_train_sm = smote.fit_resample(X2_train, y2_train)

rf2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf2.fit(X2_train_sm, y2_train_sm)
y2_prob_rf = rf2.predict_proba(X2_test)[:, 1]

mlp2 = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                     max_iter=500, random_state=42)
mlp2.fit(X2_train_sm, y2_train_sm)
y2_prob_mlp = mlp2.predict_proba(X2_test)[:, 1]


# two ROC plots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for y_true, y_prob, label, color in [
    (y1_test, y1_prob_rf,  "Random Forest", "#1f77b4"),
    (y1_test, y1_prob_mlp, "MLP",           "#ff7f0e"),
]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC curves - Dataset 1 (JRC)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
for y_true, y_prob, label, color in [
    (y2_test, y2_prob_rf,  "Random Forest", "#1f77b4"),
    (y2_test, y2_prob_mlp, "MLP",           "#ff7f0e"),
]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC curves - Dataset 2 (GDSI/MS)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle("Random Forest vs MLP - ROC comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()
print("Saved roc_curves.png")
