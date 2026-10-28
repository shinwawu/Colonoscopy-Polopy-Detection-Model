"""Compara YOLO e U-Net no mesmo conjunto interno ou externo."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation_core import (ROOT, carregar_casos, carregar_casos_internos, carregar_referencia,
                             executar_avaliacao, filtrar_mascara, ler_imagem)
from predict_unet import carregar_unet
from predict_yolo import carregar_yolo

DADOS = ROOT / "data/processed/Kvasir-SEG"
PESOS_YOLO = ROOT / "runs/yolo-deteccao/weights/best.pt"
PESOS_UNET = ROOT / "runs/unet-deteccao/weights/best.pt"
SEMENTE_AMOSTRA = 32


def figura_comparativa(casos, yolo, unet, saida, area_minima=500, n=8):
    """Desenha os dois modelos nas MESMAS imagens, lado a lado.

    Os números resumem, mas não mostram COMO cada um erra: a U-Net serrilha e
    fragmenta porque decide pixel a pixel, enquanto a máscara do YOLO nasce
    recortada pela caixa. Isso só aparece olhando o contorno.
    """
    import random

    comanotacao = [c for c in casos if c["mascara"]]
    if not comanotacao:
        return None
    random.Random(SEMENTE_AMOSTRA).shuffle(comanotacao)
    escolhidos = comanotacao[:n]

    colunas = min(len(escolhidos), 4)
    linhas = (len(escolhidos) + colunas - 1) // colunas
    fig, ax = plt.subplots(linhas, colunas, figsize=(4 * colunas, 4.2 * linhas), squeeze=False)
    for a in ax.ravel():
        a.axis("off")

    for a, caso in zip(ax.ravel(), escolhidos):
        bgr = ler_imagem(caso["caminho"])
        h, w = bgr.shape[:2]
        gt, _ = carregar_referencia(caso, (h, w))

        def dice(pred):
            inter = int(np.count_nonzero((pred > 0) & (gt > 0)))
            soma = int(np.count_nonzero(pred)) + int(np.count_nonzero(gt))
            return 2 * inter / soma if soma else 0.0

        vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rotulos = []
        # verde = anotacao, vermelho = U-Net, azul = YOLO
        for nome, predizer, cor in (("gt", None, (0, 200, 0)),
                                    ("unet", unet, (220, 0, 0)),
                                    ("yolo", yolo, (0, 90, 255))):
            if predizer is None:
                mascara = gt
            else:
                bruta, _ = predizer(bgr)
                mascara = filtrar_mascara(cv2.resize(bruta, (w, h), interpolation=cv2.INTER_NEAREST),
                                          area_minima)
                rotulos.append(f"{nome} {dice(mascara):.3f}")
            cont, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cont, -1, cor, 3)
        a.imshow(vis)
        a.set_title("   ".join(rotulos), fontsize=10)

    fig.suptitle("verde = anotacao     vermelho = U-Net     azul = YOLO", fontsize=12)
    fig.tight_layout()
    destino = Path(saida) / "comparacao_visual.png"
    fig.savefig(destino, dpi=110)
    plt.close(fig)
    return destino


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    entrada = parser.add_mutually_exclusive_group()
    entrada.add_argument("--origem", type=Path, help="Pasta/imagem externa; sem esse argumento usa o teste interno")
    entrada.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--dados", type=Path, default=DADOS)
    parser.add_argument("--anotacoes", type=Path)
    parser.add_argument("--pesos-yolo", type=Path, default=PESOS_YOLO)
    parser.add_argument("--pesos-unet", type=Path, default=PESOS_UNET)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limiar", type=float, default=0.5)
    parser.add_argument("--grade", type=int, default=384)
    parser.add_argument("--area-minima", type=int, default=500)
    parser.add_argument("--saida", type=Path, default=ROOT / "runs" / f"comparacao_{datetime.now():%Y%m%d_%H%M%S_%f}")
    args = parser.parse_args()
    if args.anotacoes and not args.origem:
        parser.error("--anotacoes só é usado com --origem externa")
    try:
        if args.saida.exists():
            raise FileExistsError(f"Saída já existe: {args.saida}")
        conjunto = "externo" if args.origem else f"interno/{args.split}"
        casos = carregar_casos(args.origem, args.anotacoes) if args.origem else carregar_casos_internos(args.dados, args.split)
        for caso in casos:
            carregar_referencia(caso, ler_imagem(caso["caminho"]).shape[:2])
        # Carrega ambos antes de produzir resultados, para detectar pesos ausentes cedo.
        yolo, cy = carregar_yolo(args.pesos_yolo, args.yolo_imgsz, args.conf, args.device)
        unet, cu = carregar_unet(args.pesos_unet, args.device, args.limiar)
        resultados = {}
        for nome, predizer, config in (("yolo", yolo, cy), ("unet", unet, cu)):
            resultados[nome] = executar_avaliacao(casos, predizer, args.saida / nome,
                                                {**config, "conjunto": conjunto}, args.grade, args.area_minima)
        chaves = lambda r: [(e["imagem"], e["sha256"], e["mascara_sha256"], e["presenca"]) for e in r["entradas"]]
        if chaves(resultados["yolo"]) != chaves(resultados["unet"]):
            raise ValueError("As imagens/anotações mudaram entre as execuções")
        resumo = [{"modelo": nome, **r["resumo"]} for nome, r in resultados.items()]
        with (args.saida / "comparacao.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(resumo[0]))
            writer.writeheader()
            writer.writerows(resumo)
        (args.saida / "comparacao.json").write_text(json.dumps({
            "comparacao_completa": True, "conjunto": conjunto, "resumo": resumo,
            "resultados": {nome: f"{nome}/resultado.json" for nome in resultados},
        }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        figura = figura_comparativa(casos, yolo, unet, args.saida, args.area_minima)
    except (OSError, ValueError) as erro:
        parser.exit(2, f"Erro: {erro}\n")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    if figura:
        print("Figura comparativa:", figura)
    print("Resultados:", args.saida.resolve())


if __name__ == "__main__":
    main()
