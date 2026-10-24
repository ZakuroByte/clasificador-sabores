# Clasificador de sabores en comida

Modelo de aprendizaje automático que, a partir de una foto de comida, predice si es **ácida, amarga, dulce, picante y/o salada**. Es una clasificación **multi-etiqueta**: un mismo platillo puede tener varios sabores a la vez (por ejemplo, dulce y ácido).

Hecho con scikit-learn, entrenado con el dataset [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) e interfaz web con Gradio.

## Requisitos

- Python 3.12
- **uv** ([instalación](https://docs.astral.sh/uv/); en Arch: `sudo pacman -S uv`) **o** **conda** (Anaconda o Miniconda). Elige solo uno.

## Instalación

Primero clona el repositorio:

```bash
git clone https://github.com/ZakuroByte/clasificador-sabores.git
cd clasificador-sabores
```

### Opción A: con uv

```bash
uv sync
```

Esto crea la carpeta `.venv` e instala las versiones exactas de `uv.lock`. No necesitas activar nada: los comandos se ejecutan con `uv run`.

### Opción B: con conda

```bash
conda create -n sabores python=3.12 -y
conda activate sabores
pip install pandas numpy scikit-learn joblib "pillow<13" "gradio>=5"
```

Cada vez que abras una terminal nueva, activa el entorno con `conda activate sabores` antes de correr los scripts.

## Probar la app (sin entrenar)

El repositorio ya incluye el modelo entrenado (`modelo.joblib`), así que no necesitas descargar nada más.

| uv | conda (con el entorno activado) |
|---|---|
| `uv run python app.py` | `python app.py` |

Abre en el navegador la dirección que imprima la terminal (normalmente `http://127.0.0.1:7860`), arrastra una foto de comida y verás la probabilidad de cada sabor.

## Reentrenar el modelo (opcional)

1. Descarga y descomprime Food-101 dentro de esta carpeta (pesa ~5 GB):

   ```bash
   wget -c https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   tar -xzf food-101.tar.gz
   ```

2. Genera el CSV de etiquetas (usa el diccionario plato → sabor de `crear_csv.py`):

   | uv | conda |
   |---|---|
   | `uv run python crear_csv.py` | `python crear_csv.py` |

3. Entrena y compara los modelos (la primera vez tarda unos minutos por la extracción de características):

   | uv | conda |
   |---|---|
   | `uv run python entrenar.py` | `python entrenar.py` |

   Esto guarda el mejor modelo en `modelo.joblib`.

## Cómo funciona

1. **Etiquetas:** Food-101 trae las fotos agrupadas por platillo (101 categorías). Se asignó manualmente uno o varios sabores a cada categoría y todas sus fotos heredan esas etiquetas (100 fotos por categoría, ~10,100 imágenes).
2. **Características:** cada foto se reduce a 32x32 píxeles y se combina con un histograma de color (16 intervalos por canal RGB).
3. **Modelos comparados:** `MLPClassifier`, `RandomForestClassifier` y `LogisticRegression` (uno por sabor). Los modelos devuelven una probabilidad por sabor (activación sigmoide) y se considera detectado el sabor cuya probabilidad supera el umbral.
4. **Evaluación:** división 80/20 y métrica F1 por sabor.

## Resultados

| Modelo | Ácido | Amargo | Dulce | Picante | Salado | Promedio |
|---|---|---|---|---|---|---|
| MLP | 0.165 | 0.103 | 0.367 | 0.104 | 0.824 | 0.313 |
| Random Forest | 0.000 | 0.000 | 0.168 | 0.000 | 0.871 | 0.208 |
| Regresión logística | 0.191 | 0.127 | 0.358 | 0.134 | 0.743 | 0.311 |

El mejor modelo (MLP) es el que se guarda y usa la app.

## Limitaciones

- **Clases desbalanceadas:** hay 7,700 fotos saladas contra solo 300 amargas, por lo que el modelo detecta mucho mejor "salado" que "amargo".
- **El sabor no siempre se ve en una foto:** una salsa roja puede ser dulce o muy picante. El modelo aprende correlaciones visuales (color, forma general), no el sabor real.
- **Etiquetas por platillo, no por foto:** el modelo en la práctica aprende a reconocer qué platillo parece y de ahí deduce el sabor. Las etiquetas también son subjetivas.
- **Características simples:** píxeles crudos y color no capturan textura ni formas. Una CNN con transfer learning (por ejemplo ResNet) mejoraría mucho los resultados.

## Posibles mejoras

- Bajar el umbral de decisión de 0.5 a 0.3 para detectar más sabores poco frecuentes.
- Agregar características de textura.
- Balancear clases o conseguir más ejemplos de "amargo" y "picante".
- Usar una red convolucional preentrenada.

## Estructura del proyecto

```
├── app.py            # Interfaz web (Gradio)
├── crear_csv.py      # Genera etiquetas.csv a partir de Food-101
├── entrenar.py       # Entrena, compara y guarda el mejor modelo
├── utils.py          # Extracción de características y utilidades compartidas
├── etiquetas.csv     # Etiquetas de sabor por imagen
├── modelo.joblib     # Modelo entrenado
├── pyproject.toml    # Dependencias
└── uv.lock           # Versiones exactas
```

`food-101/` y `features.npy` no se suben al repositorio por su tamaño.

## Trabajo en equipo

Cada integrante trabaja en su propia rama y abre un Pull Request:

```bash
git checkout -b mi-mejora
git add .
git commit -m "Describe el cambio"
git push -u origin mi-mejora
```
