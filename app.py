import gradio as gr
import joblib
from utils import SABORES, extraer, probas

modelo = joblib.load("modelo.joblib")["modelo"]

def predecir(ruta):
    if ruta is None:
        return None, ""
    p = probas(modelo, extraer(ruta).reshape(1, -1))[0]
    detectados = [s for s, v in zip(SABORES, p) if v > 0.5]
    texto = ", ".join(detectados) if detectados else "Ningún sabor superó el 50%"
    return {s: float(v) for s, v in zip(SABORES, p)}, texto

demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(type="filepath", label="Sube una foto de comida"),
    outputs=[gr.Label(num_top_classes=5, label="Probabilidad por sabor"),
             gr.Textbox(label="Sabores detectados")],
    title="Clasificador de sabores",
)
demo.launch()
