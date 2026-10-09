# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# %%
df = pd.read_csv(
    r"cumulative_2026.03.14_07.05.16.csv",
    skiprows=53,
    low_memory=False
)

df.drop(columns=[    #dropping unnecessary columns
"kepid",
"kepoi_name",
"kepler_name",
"ra",
"dec",
"koi_tce_delivname",
"koi_score",
"koi_fpflag_nt",
"koi_fpflag_ss",
"koi_fpflag_co",
"koi_fpflag_ec",
"koi_teq_err1",
"koi_teq_err2"
], inplace=True)

df.head()

df.fillna(df.mean(numeric_only=True), inplace=True) #filling missing value with mean of column
df.isnull().sum()#checking for null values after filling
df.shape  

# %%
df["koi_disposition"].value_counts()
plt.hist(np.log1p(df["koi_period"]),bins=60,color="yellow",edgecolor="black")
plt.title("Log Orbital Period Distribution")
plt.xlabel("log(1 + Orbital Period)")
plt.ylabel("Frequency")
plt.savefig(PLOT_DIR / "orbital_period_distribution.png", dpi=150, bbox_inches="tight")
plt.close()


# %%
df["label"]=df["koi_disposition"].apply(lambda x: 1 if (x=="CONFIRMED" or x=="CANDIDATE") else 0)
df.drop(columns=["koi_disposition","koi_pdisposition"],inplace=True)

# %%
X=df.drop(columns=["label"])
y=df["label"]

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=18
)

# parameter search
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=22, n_jobs=2)

grid_search = RandomizedSearchCV(
    rf,
    param_grid,
    n_iter=4,
    cv=3,
    scoring="accuracy",
    n_jobs=2,
    pre_dispatch=2,
    random_state=22
)

grid_search.fit(X_train, y_train)

# best model
best_model = grid_search.best_estimator_

# predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

# %%
from xgboost import XGBClassifier

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=2
)

xgb_model.fit(X_train, y_train)

# Generate predictions
y_pred_xgb = xgb_model.predict(X_test)

# Generate probabilities (needed for ROC)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]

# %%

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# -----------------------------
# Plot style
# -----------------------------

sns.set_theme(style="whitegrid", context="talk")

fig, axes = plt.subplots(3, 3, figsize=(20,16))
fig.suptitle("Exoplanet Detection Model Evaluation", fontsize=24, fontweight="bold")

# ---------------------------------------------------
# Get prediction probabilities
# ---------------------------------------------------

y_prob = best_model.predict_proba(X_test)[:,1]

# ---------------------------------------------------
# 1. Confusion Matrix
# ---------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="crest",
    linewidths=1,
    cbar=False,
    ax=axes[0,0]
)

axes[0,0].set_title("Confusion Matrix")
axes[0,0].set_xlabel("Predicted")
axes[0,0].set_ylabel("Actual")

# ---------------------------------------------------
# 2. ROC Curve
# ---------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

axes[0,1].plot(fpr, tpr, lw=3, color="navy", label=f"AUC = {roc_auc:.2f}")
axes[0,1].plot([0,1],[0,1],'--', color="gray")

axes[0,1].set_title("ROC Curve")
axes[0,1].set_xlabel("False Positive Rate")
axes[0,1].set_ylabel("True Positive Rate")
axes[0,1].legend()

# ---------------------------------------------------
# 3. Precision-Recall Curve
# ---------------------------------------------------

precision, recall, _ = precision_recall_curve(y_test, y_prob)

axes[0,2].plot(recall, precision, lw=3, color="darkgreen")

axes[0,2].set_title("Precisionâ€“Recall Curve")
axes[0,2].set_xlabel("Recall")
axes[0,2].set_ylabel("Precision")

# ---------------------------------------------------
# 4. Feature Importance
# ---------------------------------------------------

importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
)

top_features = importance.sort_values(ascending=False).head(10)

top_features.plot.bar(
    ax=axes[1,0],
    color="steelblue"
)

axes[1,0].set_title("Top Important Features")
axes[1,0].tick_params(axis="x", rotation=45)

# ---------------------------------------------------
# 5. Log Planet Radius Distribution
# ---------------------------------------------------

sns.histplot(
    np.log1p(df["koi_prad"]),
    bins=50,
    kde=True,
    color="purple",
    ax=axes[1,1]
)

axes[1,1].set_title("Log Planet Radius Distribution")
axes[1,1].set_xlabel("log(1 + Planet Radius)")

# ---------------------------------------------------
# 6. Log Orbital Period Distribution
# ---------------------------------------------------

sns.histplot(
    np.log1p(df["koi_period"]),
    bins=50,
    kde=True,
    color="orange",
    ax=axes[1,2]
)

axes[1,2].set_title("Log Orbital Period Distribution")
axes[1,2].set_xlabel("log(1 + Orbital Period)")

# ---------------------------------------------------
# 7. Log Transit Depth Distribution
# ---------------------------------------------------

sns.histplot(
    np.log1p(df["koi_depth"]),
    bins=50,
    kde=True,
    color="teal",
    ax=axes[2,0]
)

axes[2,0].set_title("Log Transit Depth Distribution")
axes[2,0].set_xlabel("log(1 + Transit Depth)")

# ---------------------------------------------------
# 8. Feature Correlation
# ---------------------------------------------------

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr.iloc[:10,:10],
    cmap="coolwarm",
    square=True,
    cbar=False,
    ax=axes[2,1]
)

axes[2,1].set_title("Feature Correlation")

# ---------------------------------------------------
# 9. Class Distribution
# ---------------------------------------------------

sns.countplot(
    x="label",
    data=df,
    palette="Set2",
    ax=axes[2,2]
)

axes[2,2].set_title("Class Distribution")

# ---------------------------------------------------
# Layout
# ---------------------------------------------------

plt.tight_layout()
plt.savefig(PLOT_DIR / "model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

# Random Forest performance
rf_accuracy = accuracy_score(y_test, y_pred)
rf_auc = roc_auc_score(y_test, y_prob)

# XGBoost performance
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_prob_xgb)

comparison = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [rf_accuracy, xgb_accuracy],
    "ROC_AUC": [rf_auc, xgb_auc]
})

print("Model Performance Comparison")
print(comparison)

# %%
from sklearn.metrics import confusion_matrix, roc_curve, auc

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="crest",
    linewidths=1.5,
    cbar=False
)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(PLOT_DIR / "xgboost_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()


# -----------------------------
# ROC Curve
# -----------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, lw=3, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')

plt.title("XGBoost ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig(PLOT_DIR / "xgboost_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()


# SHAP is optional because explainability on the full test set can use
# substantial memory. Install it separately to enable these plots.
try:
    import shap
except ImportError:
    print("SHAP plots skipped; install shap to enable explainability.")
else:
    shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(shap_sample)

    shap.summary_plot(
        shap_values,
        shap_sample,
        plot_type="bar",
        max_display=10,
        show=False
    )
    plt.close()

    shap.summary_plot(
        shap_values,
        shap_sample,
        max_display=15,
        show=False
    )
    plt.close()

# %%
import joblib

joblib.dump(xgb_model, "exoplanet_model.pkl")

print("Model saved successfully")

# %%
import matplotlib.pyplot as plt
import numpy as np

light_curve = X.iloc[0].values   # first star light curve
time = np.arange(len(light_curve))  # time steps

plt.figure(figsize=(12,4))
plt.plot(time, light_curve)

plt.title("Example Stellar Light Curve")
plt.xlabel("Time Step")
plt.ylabel("Brightness")

plt.tight_layout()
plt.savefig(PLOT_DIR / "example_light_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# %%
def predict_from_user_input(model, features):
    print("\nEnter one value for each feature in this order:")
    for feature in features.columns:
        print(
            f"{feature}: "
            f"{features[feature].min():.6g} to {features[feature].max():.6g}"
        )

    values_text = input(
        "\nEnter the feature values as comma-separated numbers: "
    )
    values = [float(value.strip()) for value in values_text.split(",")]

    if len(values) != len(features.columns):
        raise ValueError(
            f"Expected {len(features.columns)} values, "
            f"but received {len(values)}."
        )

    user_features = pd.DataFrame([values], columns=features.columns)
    prediction = model.predict(user_features)[0]
    print("Planet exists" if prediction == 1 else "Planet does not exist")


predict_from_user_input(xgb_model, X)