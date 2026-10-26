"""Compara YOLO e U-Net no mesmo conjunto interno ou externo."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from avaliacao_externa import carregar_casos, carregar_casos_internos, carregar_referencia, executar_avaliacao, ler_imagem
from config import DADOS, PESOS_UNET, PESOS_YOLO, ROOT
from prediz_unet import carregar_unet
from prediz_yolo import carregar_yolo


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
    except (OSError, ValueError) as erro:
        parser.exit(2, f"Erro: {erro}\n")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("Resultados:", args.saida.resolve())


if __name__ == "__main__":
    main()
