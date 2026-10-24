import numpy as np
from PIL import Image

SABORES = ["acido", "amargo", "dulce", "picante", "salado"]

def extraer(ruta):
    img = Image.open(ruta).convert("RGB").resize((32, 32))
    arr = np.array(img)
    pixeles = arr.flatten() / 255.0
    hist = np.concatenate([
        np.histogram(arr[:, :, c], bins=16, range=(0, 256))[0] for c in range(3)
    ])
    hist = hist / hist.sum()
    return np.concatenate([pixeles, hist])

def probas(modelo, X):
    """Devuelve matriz (n_imagenes x 5) con la probabilidad de cada sabor."""
    p = modelo.predict_proba(X)
    if isinstance(p, list):  # RandomForest y MultiOutput devuelven una lista
        p = np.column_stack([m[:, 1] for m in p])
    return p
