import argparse
import json
from datetime import datetime
from pathlib import Path
from evaluation_core import ROOT, dispositivo, sha256

# ---- configuracao deste modelo, toda neste arquivo ----
path = ROOT
runs_path = path / "runs"
artifacts_path = path / "artifacts"
yolo_data_path = ROOT / "data/processed/Kvasir-SEG"
NOME = "yolo-deteccao"
SEED = 42
MODELO = "yolo11s-seg.pt"
EPOCAS, IMGSZ, BATCH = 100, 640, 8

def main(argv=None):
    parser = argparse.ArgumentParser(description="Treina YOLO11-seg e avalia o melhor checkpoint no teste interno")
    parser.add_argument("--dados", type=Path, default=yolo_data_path / "data.yaml")
    parser.add_argument("--runs", type=Path, default=runs_path,
                        help="Pasta de saída dos treinos, checkpoints e avaliação")
    parser.add_argument("--artifacts", type=Path, default=artifacts_path,
                        help="Pasta de saída dos relatórios JSON")
    parser.add_argument("--nome", default=NOME, help="Nome de uma nova execução")
    parser.add_argument("--modelo", default=MODELO)
    parser.add_argument("--epocas", type=int, default=EPOCAS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--device", default="auto", help="auto, 0 para GPU ou cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=["ram", "disk", "off"], default="off")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Precisão mista; --no-amp desliga se a GPU não for compatível")
    args = parser.parse_args(argv)
    if min(args.epocas, args.imgsz, args.batch) <= 0 or args.imgsz % 32 or args.workers < 0:
        parser.error("Épocas e batch devem ser positivos, imgsz múltiplo de 32, workers >=0")
    try:
        args.device = str(dispositivo(args.device))
    except ValueError as erro:
        parser.error(str(erro))
    if not args.dados.is_file():
        parser.error("data.yaml não encontrado; prepare os dados com src/data_process.py")
    if Path(args.nome).name != args.nome or args.nome in {".", "..", ""}:
        parser.error("--nome deve ser um nome de pasta simples")
    if any(p.exists() for p in (args.runs / args.nome, args.runs / f"{args.nome}_teste",
                                args.artifacts / f"{args.nome}.json")):
        parser.error("Esse experimento já existe; escolha outro --nome para preservar os resultados")

    from ultralytics import YOLO

    model = YOLO(args.modelo)
    model.train(
        data=str(args.dados.resolve()), project=str(args.runs), name=args.nome,
        epochs=args.epocas, imgsz=args.imgsz, batch=args.batch, device=args.device,  # imgsz: tamanho da imagem de entrada
        amp=args.amp,
        patience=25,  # numero de epocas sem melhora para parar o treinamento
        cache=False if args.cache == "off" else args.cache,
        seed=args.seed, workers=args.workers,
        # flipud/fliplr: probabilidade de flip vertical e horizontal; degrees: rotacao aleatoria em graus
        flipud=0.5, fliplr=0.5, degrees=15,
    )
    # Usa o caminho real produzido pelo treinador, sem reconstruir um caminho presumido.
    pesos = Path(model.trainer.best).resolve()
    if not pesos.is_file():
        raise FileNotFoundError(f"O treinamento não produziu o checkpoint esperado: {pesos}")
    # o Ultralytics sempre grava um last.pt ao lado; guardamos somente o melhor
    ultimo = pesos.with_name("last.pt")
    if ultimo.is_file():
        ultimo.unlink()
    melhor = YOLO(str(pesos))
    metric = melhor.val(data=str(args.dados.resolve()), split="test", device=args.device,
                        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                        project=str(args.runs), name=f"{args.nome}_teste")
    manifest = args.dados.parent / "split_manifest.json"
    resultado = {
        "nome": args.nome,
        "data": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "modelo": args.modelo, "epocas_solicitadas": args.epocas,
            "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
            "amp": args.amp,
            "workers": args.workers, "cache": args.cache, "seed": args.seed,
            "split_manifest_sha256": sha256(manifest) if manifest.is_file() else None,
            "split": "test", "dados": str(args.dados.resolve()), "pesos": str(pesos),
        },
        # .map E o mAP50-95 (nao existe atributo map50_95); .map50 e o mAP no IoU 0.5
        "metricas": {
            "box_map50": metric.box.map50, "box_map50_95": metric.box.map,
            "seg_map50": metric.seg.map50, "seg_map50_95": metric.seg.map,
            "precisao": metric.box.mp, "recall": metric.box.mr,
        },
    }
    # salvar metricas na pasta artifacts, uma por experimento (nao sobrescreve as anteriores).
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / f"{args.nome}.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(json.dumps(resultado, indent=2))


if __name__ == "__main__":
    main()
