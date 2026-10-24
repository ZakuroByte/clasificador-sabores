"""Entrena con los embeddings de ResNet50 (solo sklearn, no necesita torch ni GPU).

Compara dos enfoques:
  A) platillo -> sabor  (clasificador de 101 platillos y luego P(sabor) = P(platillo) @ M)
  B) directo            (una regresión logística por sabor)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, accuracy_score
from utils import SABORES

df = pd.read_csv("etiquetas.csv")
X = np.load("embeddings_resnet50.npy")
assert len(X) == len(df), "embeddings y etiquetas.csv no coinciden: vuelve a correr extraer_embeddings.py"

platos = df["plato"].values
y = df[SABORES].values

idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=platos)

# ---------- A) platillo -> sabor ----------
print("Entrenando A) clasificador de platillos...")
modelo_A = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))
modelo_A.fit(X[tr], platos[tr])
print("Accuracy de platillo:", round(accuracy_score(platos[te], modelo_A.predict(X[te])), 3))

clases = modelo_A.classes_
M = df.groupby("plato")[SABORES].first().loc[clases].values.astype(float)
P_A = np.clip(modelo_A.predict_proba(X[te]) @ M, 0, 1)

# ---------- B) directo ----------
print("Entrenando B) un clasificador por sabor...")
modelo_B = make_pipeline(
    StandardScaler(),
    MultiOutputClassifier(LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")),
)
modelo_B.fit(X[tr], y[tr])
P_B = np.column_stack([p[:, 1] for p in modelo_B.predict_proba(X[te])])

# ---------- Comparación ----------
filas = {}
for nombre, P in (("A platillo->sabor", P_A), ("B directo", P_B)):
    for umbral in (0.5, 0.3):
        pred = (P > umbral).astype(int)
        filas[f"{nombre} (u={umbral})"] = f1_score(y[te], pred, average=None, zero_division=0)

tabla = pd.DataFrame(filas, index=SABORES).T
tabla["promedio"] = tabla.mean(axis=1)
print("\n=== F1 por sabor ===")
print(tabla.round(3))
print("\nReferencias anteriores: MLP con píxeles 0.313 | platillo->sabor con píxeles 0.336")

joblib.dump({"modelo": modelo_A, "M": M, "clases": clases}, "modelo_embeddings_A.joblib")
joblib.dump({"modelo": modelo_B}, "modelo_embeddings_B.joblib")
print("\nGuardados: modelo_embeddings_A.joblib y modelo_embeddings_B.joblib")
