"""Caminhos e pré-processamento compartilhados pelo treino e pela inferência."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "data/processed/Kvasir-SEG"
NOME_YOLO = "yolo-polyp"
NOME_UNET = "unet-polyp"
PESOS_YOLO = ROOT / "runs" / NOME_YOLO / "weights/best.pt"
PESOS_UNET = ROOT / "runs" / NOME_UNET / "weights/best.pt"
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)


def dispositivo(valor="auto"):
    import torch

    if valor == "auto":
        valor = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(valor).isdigit():
        valor = f"cuda:{valor}"
    device = torch.device(valor)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponível neste ambiente; confira a instalação ou use --device cpu")
    return device
