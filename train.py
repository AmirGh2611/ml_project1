import numpy as np
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1) load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2) split into training and testing sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) model
model = LogisticRegression(max_iter=200)

# 5) cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\n=== {model} ===")
print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# training on the entire training set
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 6) save model + scaler together
os.makedirs("models", exist_ok=True)
path = "models/model.joblib"
joblib.dump({"scaler": scaler, "model": model}, path)
print(f"Saved: {path}")