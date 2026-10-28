"""Predição U-Net em imagens externas"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluation_core import ROOT, carregar_casos, dispositivo, executar_avaliacao, sha256
# a normalizacao vem gravada no checkpoint; estas constantes so atendem o
# state_dict puro do baseline original, que nao carrega configuracao junto
from train_unet import DESVIO, MEDIA
from unet import UNet

PESOS_UNET = ROOT / "runs/unet-deteccao/weights/best.pt"


def carregar_unet(pesos, device="cpu", limiar=0.5):
    """Retorna predição binária na grade nativa e sua configuração de avaliação."""
    pesos = Path(pesos).resolve()
    if not pesos.is_file():
        raise FileNotFoundError(f"Checkpoint U-Net não encontrado: {pesos}")
    if not 0 <= limiar <= 1:
        raise ValueError("O limiar deve estar entre 0 e 1")
    device = dispositivo(device)
    checkpoint = torch.load(pesos, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        config = checkpoint["config"]
        state = checkpoint["state_dict"]
    else:
        # Compatibilidade com o state_dict puro do baseline original.
        config, state = {"base": 32, "imgsz": 384}, checkpoint
    base, imgsz = int(config["base"]), int(config["imgsz"])
    media = np.array(config.get("media", MEDIA), np.float32)
    desvio = np.array(config.get("desvio", DESVIO), np.float32)
    modelo = UNet(n_channels=3, n_classes=1, base=base).to(device)
    modelo.load_state_dict(state)
    modelo.eval()

    @torch.inference_mode()
    def predizer(imagem):
        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        normalizada = (rgb.astype(np.float32) / 255.0 - media) / desvio
        x = torch.from_numpy(np.ascontiguousarray(normalizada)).permute(2, 0, 1)[None].to(device)
        prob = torch.sigmoid(modelo(x))[0, 0].cpu().numpy()
        mascara = (prob > limiar).astype(np.uint8)
        return mascara, {}

    return predizer, {"modelo": "unet", "pesos": str(pesos), "pesos_sha256": sha256(pesos),
                      "imgsz": imgsz, "base": base, "limiar": limiar, "device": str(device),
                      "formato_predicao": [imgsz, imgsz],
                      "media": media.tolist(), "desvio": desvio.tolist(), "torch": str(torch.__version__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origem", required=True, type=Path)
    parser.add_argument("--anotacoes", type=Path)
    parser.add_argument("--pesos", type=Path, default=PESOS_UNET)
    parser.add_argument("--saida", type=Path, default=ROOT / "runs" / f"unet_externo_{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--limiar", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grade", type=int, default=384)
    parser.add_argument("--area-minima", type=int, default=500)
    args = parser.parse_args()
    try:
        casos = carregar_casos(args.origem, args.anotacoes)
        predizer, config = carregar_unet(args.pesos, args.device, args.limiar)
        resultado = executar_avaliacao(casos, predizer, args.saida, config, args.grade, args.area_minima)
    except (OSError, ValueError) as erro:
        parser.exit(2, f"Erro: {erro}\n")
    print(json.dumps(resultado["resumo"], ensure_ascii=False, indent=2))
    print(f"Resultados salvos em {args.saida.resolve()}")


if __name__ == "__main__":
    main()
