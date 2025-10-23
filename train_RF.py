import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# === Load and merge all CSV files ===
csv_files = glob.glob("/home/j4_m3s/Documents/uni_ws/image_processing/mini_project/csv_folder/fall_data_*.csv")
print(f"🔍 Found {len(csv_files)} CSV files")

if len(csv_files) == 0:
    raise FileNotFoundError("❌ No fall_data_*.csv files found!")

df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
print(f"✅ Combined dataset shape: {df.shape}")

# === Preprocessing ===
# Remove duplicates, missing values if any
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Features and Labels
X = df[["cx", "cy", "width", "height", "aspect_ratio", "tilt_angle", "vertical_speed"]]
y = df["label"]

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Train Random Forest Model ===
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# === Evaluate Model ===
y_pred = rf.predict(X_test)
print("\n=== 📊 Evaluation Results ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === Feature Importance ===
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.title("Feature Importance (Random Forest)")
plt.barh(range(len(features)), importances[indices][::-1])
plt.yticks(range(len(features)), features[indices][::-1])
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# === Save Model ===
joblib.dump(rf, "human_fall_model.pkl")
print("\n✅ Model saved")