# Clasificador de sabores en comida

Modelo de aprendizaje automático que, a partir de una foto de comida, predice si es **ácida, amarga, dulce, picante y/o salada**. Es una clasificación **multi-etiqueta**: un mismo platillo puede tener varios sabores a la vez (por ejemplo, dulce y ácido).

Usa [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz), una red **ResNet50 preentrenada** como extractor de características, **scikit-learn** para clasificar e interfaz web con **Gradio**.

## Requisitos

- Python 3.12
- **uv** ([instalación](https://docs.astral.sh/uv/); en Arch: `sudo pacman -S uv`) **o** **conda** (Anaconda o Miniconda). Elige solo uno.
- ~1 GB de espacio para torch (versión CPU) y los pesos de ResNet50.

## Instalación

Primero clona el repositorio:

```bash
git clone https://github.com/ZakuroByte/clasificador-sabores.git
cd clasificador-sabores
```

### Opción A: con uv

```bash
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> Después de instalar torch, **no vuelvas a correr `uv sync`**: borraría torch del entorno. Con `uv run` no hay problema. Si lo corres por error, repite el segundo comando.

### Opción B: con conda

```bash
conda create -n sabores python=3.12 -y
conda activate sabores
pip install pandas numpy scikit-learn joblib "pillow<13" "gradio>=5"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Cada vez que abras una terminal nueva, activa el entorno con `conda activate sabores`.

## Probar la app (sin entrenar)

El repositorio ya incluye el modelo entrenado (`modelo_embeddings_A.joblib`).

| uv | conda (con el entorno activado) |
|---|---|
| `uv run python app.py` | `python app.py` |

Abre en el navegador la dirección que imprima la terminal (normalmente `http://127.0.0.1:7860`) y arrastra una foto de comida. La app muestra la probabilidad de cada sabor, los 3 platillos más parecidos y los sabores detectados (probabilidad mayor a 0.5).

La primera vez, torchvision descarga los pesos de ResNet50 (~100 MB), y se guardan para las siguientes. Cada predicción tarda menos de un segundo en CPU.

## Reentrenar el modelo (opcional)

1. Descarga y descomprime Food-101 dentro de esta carpeta (~5 GB):

   ```bash
   wget -c https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   tar -xzf food-101.tar.gz
   ```

2. Genera el CSV de etiquetas (300 fotos por platillo, ~30,300 imágenes):

   | uv | conda |
   |---|---|
   | `uv run python crear_csv.py` | `python crear_csv.py` |

3. Extrae los vectores con ResNet50 (se corre una sola vez). Con GPU NVIDIA tarda unos minutos; solo con CPU, entre 20 y 40 minutos. Genera `embeddings_resnet50.npy` (~250 MB, no se sube al repo).

   | uv | conda |
   |---|---|
   | `uv run python extraer_embeddings.py` | `python extraer_embeddings.py` |

4. Entrena y compara (solo sklearn, tarda un par de minutos). Guarda `modelo_embeddings_A.joblib`, que usa la app.

   | uv | conda |
   |---|---|
   | `uv run python entrenar_embeddings.py` | `python entrenar_embeddings.py` |

## Cómo funciona

1. **Etiquetas:** Food-101 trae las fotos agrupadas por platillo (101 categorías). Se asignó manualmente uno o varios sabores a cada categoría (diccionario en `crear_csv.py`), y todas sus fotos heredan esas etiquetas.
2. **Características:** ResNet50 preentrenado en ImageNet (sin su capa final) convierte cada foto en un vector de 2048 números. No se entrena la red, solo se usa para extraer características.
3. **Enfoque A (el que usa la app), platillo → sabor:** una regresión logística multiclase (softmax) predice cuál de los 101 platillos es. Luego la probabilidad de cada sabor se calcula sumando las probabilidades de los platillos que tienen ese sabor: `P(sabor) = P(platillo) @ M`, donde `M` es la matriz platillo-sabor.
4. **Enfoque B, directo:** una regresión logística por sabor, como comparación.
5. **Evaluación:** división 80/20 estratificada por platillo y F1 por sabor.

## Resultados

F1 por sabor en el conjunto de prueba:

| Modelo | Ácido | Amargo | Dulce | Picante | Salado | Promedio |
|---|---|---|---|---|---|---|
| **A) ResNet50 + platillo → sabor (u=0.5)** | 0.743 | 0.582 | 0.829 | 0.673 | 0.956 | **0.757** |
| B) ResNet50 + un clasificador por sabor (u=0.5) | 0.581 | 0.493 | 0.747 | 0.464 | 0.919 | 0.641 |
| Platillo → sabor con píxeles 32x32 + color (u=0.3) | 0.166 | 0.124 | 0.481 | 0.043 | 0.866 | 0.336 |
| MLP con píxeles 32x32 + color (u=0.5) | 0.165 | 0.103 | 0.367 | 0.104 | 0.824 | 0.313 |

La accuracy del clasificador de platillos pasó de 7.1% (píxeles y color) a 59.9% (ResNet50). El salto principal viene de la calidad de las características, no del clasificador.

Los scripts `entrenar.py` y `entrenar_platillo.py` corresponden a los primeros enfoques (píxeles y color) y se conservan como punto de comparación.

## Limitaciones

- **Solo conoce 101 platillos.** Una foto de algo fuera de Food-101 (una toronja, un café) se interpreta como el platillo más parecido de los 101.
- **Evaluación optimista:** la división es por foto, no por platillo, así que el modelo ya vio los platillos de prueba durante el entrenamiento. Con platillos totalmente nuevos, el resultado sería menor.
- **Etiquetas por platillo, no por foto:** todas las fotos de un platillo comparten sabor, aunque en la realidad varíe (con o sin salsa, por ejemplo). Las etiquetas también son subjetivas.
- **Clases desbalanceadas:** hay muchos más platillos salados que amargos (solo chocolate y tiramisú).
- **El sabor no siempre se ve en una foto.**

## Posibles mejoras

- Agregar categorías nuevas de amargo, ácido y picante (café, toronja, limón, chiles) con fotos de otras fuentes.
- Evaluar dejando platillos completos fuera del entrenamiento.
- Probar otros extractores (EfficientNet, ViT) o hacer fine-tuning de la red.

## Estructura del proyecto

```
├── app.py                       # Interfaz web (Gradio)
├── crear_csv.py                 # Genera etiquetas.csv a partir de Food-101
├── extraer_embeddings.py        # ResNet50: foto -> vector de 2048 números
├── entrenar_embeddings.py       # Entrena y compara los enfoques A y B
├── entrenar.py                  # Primer enfoque (píxeles + color), referencia
├── entrenar_platillo.py         # Segundo enfoque (píxeles + platillo), referencia
├── utils.py                     # Extracción de características simples y constantes
├── etiquetas.csv                # Etiquetas de sabor por imagen
├── modelo_embeddings_A.joblib   # Modelo que usa la app
├── pyproject.toml               # Dependencias
└── uv.lock                      # Versiones exactas
```

`food-101/` y los archivos `.npy` no se suben al repositorio por su tamaño.

## Trabajo en equipo

Cada integrante trabaja en su propia rama y abre un Pull Request:

```bash
git checkout -b mi-mejora
git add .
git commit -m "Describe el cambio"
git push -u origin mi-mejora
```
