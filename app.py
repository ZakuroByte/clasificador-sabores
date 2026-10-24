import joblib
import numpy as np
import gradio as gr
from PIL import Image
from utils import SABORES

try:
    import torch
    from torchvision import models, transforms
except ImportError:
    raise SystemExit(
        "Falta torch. Instálalo (versión CPU) con:\n"
        "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
        "o, con conda (entorno activado):\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    )

UMBRAL = 0.5
torch.set_num_threads(4)

# Clasificador de platillos (sklearn) + matriz platillo -> sabor
datos = joblib.load("modelo_embeddings_A.joblib")
clasificador, M, clases = datos["modelo"], datos["M"], datos["clases"]

# ResNet50 preentrenado como extractor de características (sin la capa final)
extractor = models.resnet50(weights="IMAGENET1K_V2")
extractor.fc = torch.nn.Identity()
extractor.eval()

transformar = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def embedding(ruta):
    img = Image.open(ruta).convert("RGB")
    x = transformar(img).unsqueeze(0)
    with torch.inference_mode():
        return extractor(x).numpy()


def predecir(ruta):
    if ruta is None:
        return None, ""
    p_plato = clasificador.predict_proba(embedding(ruta))[0]
    p_sabor = np.clip(p_plato @ M, 0, 1)

    sabores = {s: float(v) for s, v in zip(SABORES, p_sabor)}
    detectados = [s for s, v in zip(SABORES, p_sabor) if v > UMBRAL]
    texto = ", ".join(detectados) if detectados else f"Ningún sabor superó el {int(UMBRAL * 100)}%"
    return sabores, texto


demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(type="filepath", label="Sube una foto de comida"),
    outputs=[
        gr.Label(num_top_classes=5, label="Probabilidad por sabor"),
        gr.Textbox(label="Sabores detectados"),
    ],
    title="Clasificador de sabores",
    description="Predice si una comida es ácida, amarga, dulce, picante y/o salada. "
                "Solo conoce los 101 platillos de Food-101.",
)

if __name__ == "__main__":
    demo.launch()
