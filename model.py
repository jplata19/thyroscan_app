import pandas as pd
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np

# --- Configuración inicial ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Carga y preparación de datos ---
df = pd.read_csv(os.path.join(BASE_DIR, "thyroid_cancer_risk_data.csv"))
df['Diagnosis'] = df['Diagnosis'].map({'Benign': 0, 'Malignant': 1})

# Eliminar la variable 'Thyroid_Cancer_Risk'
df = df.drop(columns=['Thyroid_Cancer_Risk'], errors='ignore')

# Codificación de variables categóricas
nominal_cols = ['Gender', 'Country', 'Ethnicity', 'Family_History', 'Radiation_Exposure',
                'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']

encoders_path = os.path.join(BASE_DIR, "encoders")
os.makedirs(encoders_path, exist_ok=True)

for col in nominal_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    joblib.dump(le, os.path.join(encoders_path, f"label_encoder_{col}.pkl"))

# Limpieza y división
df = df.drop(['Patient_ID', 'Age_Group'], axis=1, errors='ignore').dropna()
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Escalado
cols_to_scale = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
scaler = MinMaxScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))

# Guardar columnas del modelo
with open(os.path.join(BASE_DIR, "model_columns.pkl"), "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# --- Balanceo con SMOTE ---
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print("\nDistribución después de SMOTE (dataset completo):", Counter(y_smote))

# --- División en train/test ---
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, stratify=y_smote, random_state=42)

# --- Entrenamiento con XGBoost ---
xgb_model = XGBClassifier(
    n_estimators=731,
    max_depth=5,
    learning_rate=0.275803457134986,
    subsample=0.5671416511454728,
    colsample_bytree=0.8174886100368205,
    gamma=1.3549752078125734,
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False  # Advertencia relacionada con el encoder
)
xgb_model.fit(X_train, y_train)

# --- Evaluación ---
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # Obtener probabilidades para la clase "Malignant"
best_threshold = 0.44
y_pred = (y_pred_prob >= best_threshold).astype(int)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Métrica F2-score
print(f"\nF2-Score: {fbeta_score(y_test, y_pred, beta=2):.4f}")

# --- Guardar modelo con umbral ajustado ---
joblib.dump(xgb_model, os.path.join(BASE_DIR, 'xgb_diagnosis_model_smote.pkl'))

# Guardar el umbral ajustado para futuras predicciones
with open(os.path.join(BASE_DIR, 'best_threshold.pkl'), 'wb') as f:
    pickle.dump(best_threshold, f)

























    





