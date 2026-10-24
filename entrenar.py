import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from utils import SABORES, extraer, probas

RUTA = "food-101/images"

df = pd.read_csv("etiquetas.csv")

# Extraer características (se guardan para no repetirlo)
if os.path.exists("features.npy"):
    X = np.load("features.npy")
else:
    print("Extrayendo características de", len(df), "imágenes...")
    X = np.array([extraer(f"{RUTA}/{n}") for n in df["imagen"]])
    np.save("features.npy", X)
y = df[SABORES].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelos = {
    "MLP": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
    ),
    "RandomForest": make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    ),
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")),
    ),
}

resultados = {}
for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre}...")
    modelo.fit(X_train, y_train)
    pred = (probas(modelo, X_test) > 0.5).astype(int)
    f1s = f1_score(y_test, pred, average=None, zero_division=0)
    resultados[nombre] = f1s
    print({s: round(f, 3) for s, f in zip(SABORES, f1s)})
    print("F1 promedio:", round(f1s.mean(), 3))

tabla = pd.DataFrame(resultados, index=SABORES).T
tabla["promedio"] = tabla.mean(axis=1)
print("\n=== Comparación (F1 por sabor) ===")
print(tabla.round(3))

mejor = tabla["promedio"].idxmax()
print("\nMejor modelo:", mejor)
joblib.dump({"modelo": modelos[mejor], "nombre": mejor}, "modelo.joblib")
print("Guardado en modelo.joblib")
