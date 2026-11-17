
import pandas as pd
import numpy as np

df = pd.read_csv("heart_disease.csv")

print(df.head())
print(df.info())

X = df.drop(["target"], axis=1)
Y = df["target"]

# Split into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# Dice score function
def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)

    return (2 * intersection) / (
        np.sum(y_true) + np.sum(y_pred) + 1e-8
    )

#Basline Random Forest 

model = RandomForestClassifier(
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Baseline model:")

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Dice Score:", dice_score(Y_test.to_numpy(), Y_pred))
print("F1 Score:", f1_score(Y_test, Y_pred))
print("ROC-AUC:", roc_auc_score(
    Y_test, model.predict_proba(X_test)[:, 1]
))

print(classification_report(Y_test, Y_pred))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

# Check overfitting
train_accuracy = model.score(X_train, Y_train)
test_accuracy = model.score(X_test, Y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

#Finetune GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 4, 6, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 5],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, Y_train)

best_model = grid_search.best_estimator_

print("GridSearchCV")

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1:", grid_search.best_score_)

#Finetuned model predictions 

Y_pred_tuned = best_model.predict(X_test)

print("FineTuned Model")

print("Accuracy:", accuracy_score(Y_test, Y_pred_tuned))
print("Dice Score:", dice_score(Y_test.to_numpy(), Y_pred_tuned))
print("F1 Score:", f1_score(Y_test, Y_pred_tuned))
print("ROC-AUC:", roc_auc_score(
    Y_test, best_model.predict_proba(X_test)[:, 1]
))

print(classification_report(Y_test, Y_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred_tuned))

#Baseline vs Finetuned
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Dice Score", "F1 Score", "ROC-AUC"],
    "Baseline": [
        accuracy_score(Y_test, Y_pred),
        dice_score(Y_test.to_numpy(), Y_pred),
        f1_score(Y_test, Y_pred),
        roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
    ],
    "Fine-Tuned": [
        accuracy_score(Y_test, Y_pred_tuned),
        dice_score(Y_test.to_numpy(), Y_pred_tuned),
        f1_score(Y_test, Y_pred_tuned),
        roc_auc_score(Y_test, best_model.predict_proba(X_test)[:, 1])
    ]
})

print("Model Comparison")
print(comparison)

#Predictions

results = pd.DataFrame({
    "Actual": Y_test.values,
    "Baseline Prediction": Y_pred,
    "Fine-Tuned Prediction": Y_pred_tuned
})

results.to_csv("predictions.csv", index=False)

print("\nPredictions saved to predictions.csv")