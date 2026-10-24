"""Paso 1: predecir el PLATILLO (101 clases) y convertir a sabor.

P(sabor) = P(platillo) @ M, donde M[i, j] = 1 si el platillo i tiene el sabor j.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from utils import SABORES, extraer

RUTA = "food-101/images"
CACHE = "features_300.npy"   # nombre distinto para no mezclar con el cache viejo

df = pd.read_csv("etiquetas.csv")
assert "plato" in df.columns, "Falta la columna 'plato': actualiza crear_csv.py y vuélvelo a correr"

# ---- Características ----
if os.path.exists(CACHE):
    X = np.load(CACHE)
    assert len(X) == len(df), "El cache no coincide con etiquetas.csv: borra features_300.npy"
else:
    print(f"Extrayendo características de {len(df)} imágenes (tarda unos minutos)...")
    X = np.array([extraer(f"{RUTA}/{n}") for n in df["imagen"]])
    np.save(CACHE, X)

platos = df["plato"].values
y_sabor = df[SABORES].values

# ---- División estratificada por platillo ----
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=platos)

# ---- Clasificador de platillo (multiclase, softmax) ----
modelo = make_pipeline(
    StandardScaler(),
    PCA(n_components=150, random_state=42),   # acelera y reduce ruido
    LogisticRegression(max_iter=3000),
)
print("Entrenando clasificador de platillos...")
modelo.fit(X[tr], platos[tr])

pred_plato = modelo.predict(X[te])
print("Accuracy de platillo:", round(accuracy_score(platos[te], pred_plato), 3))

# ---- Matriz platillo -> sabor (en el mismo orden que modelo.classes_) ----
clases = modelo.classes_
M = (
    df.groupby("plato")[SABORES].first()   # el sabor es igual para todas las fotos de un platillo
    .loc[clases]
    .values.astype(float)
)

P_plato = modelo.predict_proba(X[te])
P_sabor = np.clip(P_plato @ M, 0, 1)

# ---- Evaluación ----
print("\n=== F1 por sabor ===")
resumen = {}
for umbral in (0.5, 0.3):
    pred = (P_sabor > umbral).astype(int)
    f1s = f1_score(y_sabor[te], pred, average=None, zero_division=0)
    resumen[f"umbral {umbral}"] = f1s
tabla = pd.DataFrame(resumen, index=SABORES).T
tabla["promedio"] = tabla.mean(axis=1)
print(tabla.round(3))
print("\nReferencia: el MLP anterior tenía promedio 0.313")

joblib.dump({"modelo": modelo, "M": M, "clases": clases}, "modelo_platillo.joblib")
print("\nGuardado en modelo_platillo.joblib")
