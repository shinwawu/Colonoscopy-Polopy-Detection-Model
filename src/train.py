import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

path = Path(__file__).parent.parent
yolo_data_path = path / "data/processed/Kvasir-SEG"
runs_path = path / "runs"
artifacts_path = path / "artifacts"


NOME = "colonoscopy-polyp-detection-model"
MODELO = "yolo11s-seg.pt"
EPOCAS, IMGSZ, BATCH = 100, 640, 8

model = YOLO(MODELO)
model.train(
    data=str(yolo_data_path / "data.yaml"),
    project=str(runs_path),
    name=NOME,
    epochs=EPOCAS,
    imgsz=IMGSZ, # tamanho da imagem de entrada
    batch=BATCH,
    device=0,
    patience=25, # numero de epocas sem melhora para parar o treinamento
    cache="ram",
    seed=42,
    workers=4,
    flipud=0.5, # probabilidade de flip vertical
    fliplr=0.5, # probabilidade de flip horizontal
    degrees=15, # rotacao aleatoria em graus
)

# o train() ja recarrega o best.pt no objeto, entao o val abaixo usa o melhor checkpoint
metric = model.val(split="test", project=str(runs_path), name=f"{NOME}_teste")

# salvar metricas na pasta artifacts, uma por experimento (nao sobrescreve as anteriores).
# .map E o mAP50-95 (nao existe atributo map50_95); .map50 e o mAP no IoU 0.5
resultado = {
    "nome": NOME,
    "data": datetime.now().isoformat(timespec="seconds"),
    "config": {
        "modelo": MODELO,
        "epocas": EPOCAS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "split": "test",
        "pesos": f"runs/{NOME}/weights/best.pt",
    },
    "metricas": {
        "box_map50": metric.box.map50,
        "box_map50_95": metric.box.map,
        "seg_map50": metric.seg.map50,
        "seg_map50_95": metric.seg.map,
        "precisao": metric.box.mp,
        "recall": metric.box.mr,
    },
}
artifacts_path.mkdir(parents=True, exist_ok=True)
(artifacts_path / f"{NOME}.json").write_text(json.dumps(resultado, indent=2))
print(json.dumps(resultado["metricas"], indent=2))
