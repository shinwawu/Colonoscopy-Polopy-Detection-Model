"""Predição YOLO em imagens externas, com métricas apenas quando há anotação."""

import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from avaliacao_externa import carregar_casos, executar_avaliacao, sha256
from config import PESOS_YOLO, ROOT, dispositivo

PESOS_PADRAO = PESOS_YOLO


def carregar_yolo(pesos, imgsz=640, conf=0.25, device="cpu"):
    """Carrega somente um checkpoint local de segmentação com uma classe."""
    pesos = Path(pesos).resolve()
    if not pesos.is_file():
        raise FileNotFoundError(f"Checkpoint treinado não encontrado: {pesos}. Informe --pesos.")
    if imgsz <= 0 or not 0 <= conf <= 1:
        raise ValueError("imgsz deve ser positivo e conf deve estar entre 0 e 1")
    device = str(dispositivo(device))
    from ultralytics import YOLO

    modelo = YOLO(str(pesos))
    if modelo.task != "segment" or len(modelo.names) != 1:
        raise ValueError("Use o YOLO de segmentação treinado para a classe pólipo")

    def predizer(imagem):
        r = modelo.predict(imagem, imgsz=imgsz, conf=conf, device=device,
                           retina_masks=True, verbose=False)[0]
        mascara = np.zeros(imagem.shape[:2], np.uint8)
        if r.masks is not None:
            # A união dos mapas raster preserva também os buracos das máscaras.
            mascara = (r.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
        confiancas = r.boxes.conf.cpu().tolist() if r.boxes is not None else []
        return mascara, {"instancias_antes_filtro": len(confiancas),
                         "confianca_maxima": max(confiancas) if confiancas else None}

    config = {"modelo": "yolo", "pesos": str(pesos), "pesos_sha256": sha256(pesos),
              "imgsz": imgsz, "conf": conf, "device": device, "retina_masks": True,
              "ultralytics": version("ultralytics")}
    return predizer, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origem", required=True, type=Path, help="Uma imagem ou pasta (busca recursiva)")
    parser.add_argument("--pesos", type=Path, default=PESOS_PADRAO)
    parser.add_argument("--anotacoes", type=Path, help="CSV imagem,mascara,presenca,origem; opcional")
    parser.add_argument("--saida", type=Path, default=ROOT / "runs" / f"yolo_externo_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="cpu ou cuda:0")
    parser.add_argument("--grade", type=int, default=384, help="Lado da grade de avaliação")
    parser.add_argument("--area-minima", type=int, default=500, help="Área mínima na grade; 0 desliga o filtro")
    args = parser.parse_args()
    try:
        casos = carregar_casos(args.origem, args.anotacoes)
        predizer, config = carregar_yolo(args.pesos, args.imgsz, args.conf, args.device)
        resultado = executar_avaliacao(casos, predizer, args.saida, config, args.grade, args.area_minima)
    except (OSError, ValueError) as erro:
        parser.exit(2, f"Erro: {erro}\n")
    print(json.dumps(resultado["resumo"], ensure_ascii=False, indent=2))
    print(f"Resultados salvos em {args.saida.resolve()}")


if __name__ == "__main__":
    main()
