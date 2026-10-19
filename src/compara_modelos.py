import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from train_unet import AREA_MINIMA, BASE, IMGSZ, KvasirSeg, avalia, remove_blobs
from unet import UNet

path = Path(__file__).parent.parent
data_path = path / "data/processed/Kvasir-SEG"
artifacts_path = path / "artifacts"

UNET_PESOS = path / "runs/unet-baseline/weights/best.pt"
YOLO_PESOS = path / "runs/s640_grupo/weights/best.pt"
# configuracoes do YOLO: tamanho da imagem e confianca minima para considerar uma deteccao
YOLO_IMGSZ, YOLO_CONF = 640, 0.25


def metricas(dices, ious, fp, n_fundo):
    d = np.array(dices)
    return {
        "dice": float(d.mean()),
        "dice_mediana": float(np.median(d)),
        "dice_erro_padrao": float(d.std(ddof=1) / np.sqrt(len(d))),
        "iou": float(np.mean(ious)),
        "fp_fundos": fp / n_fundo if n_fundo else 0.0,
        "falhas_dice_abaixo_0.5": int((d < 0.5).sum()),
        "acertos_dice_acima_0.9": int((d > 0.9).sum()),
        "n_positivos": len(dices),
        "n_fundos": n_fundo,
    }


def avalia_unet(area_minima=AREA_MINIMA):
    modelo = UNet(n_channels=3, n_classes=1, base=BASE).cuda().eval()
    modelo.load_state_dict(torch.load(UNET_PESOS))
    te = DataLoader(KvasirSeg("test"), batch_size=8, num_workers=4, pin_memory=True)
    # reaproveita o avalia() do treino, mas guardando o Dice de cada imagem para
    # poder olhar a distribuicao, e nao so a media
    dices, ious, fp, n_fundo = [], [], 0, 0
    with torch.no_grad():
        for x, y in te:
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logit = modelo(x)
            pred = (torch.sigmoid(logit.float()) > 0.5).float()
            for p_i, y_i in zip(pred, y):
                p_i = remove_blobs(p_i, area_minima)
                if y_i.sum() == 0:
                    n_fundo += 1
                    fp += int(p_i.sum() > 0)
                else:
                    inter = (p_i * y_i).sum().item()
                    soma = (p_i.sum() + y_i.sum()).item()
                    uniao = ((p_i + y_i) > 0).sum().item()
                    dices.append(2 * inter / soma if soma else 0.0)
                    ious.append(inter / uniao if uniao else 0.0)
    return metricas(dices, ious, fp, n_fundo)


def avalia_yolo(conf=YOLO_CONF, area_minima=AREA_MINIMA):
    modelo = YOLO(YOLO_PESOS)
    imgs = sorted((data_path / "images/test").glob("*.jpg"))
    masks = data_path / "masks_binary"
    dices, ious, fp, n_fundo = [], [], 0, 0
    for f in imgs:
        bgr = cv2.imread(str(f))
        h, w = bgr.shape[:2]
        r = modelo.predict(bgr, imgsz=YOLO_IMGSZ, conf=conf, verbose=False)[0]
        # o YOLO devolve uma mascara por instancia; a U-Net devolve um mapa unico.
        # para comparar, uno as instancias em uma mascara binaria so.
        pred = np.zeros((h, w), np.uint8)
        if r.masks is not None:
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(pred, [poly.astype(np.int32)], 1)
        # projeta na grade da U-Net 
        pred = cv2.resize(pred, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)
        if area_minima > 0 and pred.sum():
            n, rot, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] < area_minima:
                    pred[rot == k] = 0

        mp = masks / f"{f.stem}.png"
        gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else np.zeros((h, w), np.uint8)
        gt = (cv2.resize(gt, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

        if gt.sum() == 0:
            n_fundo += 1
            fp += int(pred.sum() > 0)
        else:
            inter = int((pred & gt).sum())
            soma, uniao = int(pred.sum() + gt.sum()), int((pred | gt).sum())
            dices.append(2 * inter / soma if soma else 0.0)
            ious.append(inter / uniao if uniao else 0.0)
    return metricas(dices, ious, fp, n_fundo)


def main():
    resultado = {
        "split": "test",
        "grade": f"{IMGSZ}x{IMGSZ}",
        "area_minima": AREA_MINIMA,
        "unet": {"pesos": str(UNET_PESOS.relative_to(path)), **avalia_unet()},
        "yolo": {"pesos": str(YOLO_PESOS.relative_to(path)), "conf": YOLO_CONF,
                 **avalia_yolo()},
    }
    artifacts_path.mkdir(exist_ok=True)
    saida = artifacts_path / "comparacao.json"
    saida.write_text(json.dumps(resultado, indent=2))

    print(f"{'':12} {'dice':>7} {'mediana':>8} {'iou':>7} {'fp_fundos':>10} {'>0.9':>5} {'<0.5':>5}")
    for nome in ("unet", "yolo"):
        m = resultado[nome]
        print(f"{nome:12} {m['dice']:7.4f} {m['dice_mediana']:8.4f} {m['iou']:7.4f} "
              f"{m['fp_fundos']:10.4f} {m['acertos_dice_acima_0.9']:5d} "
              f"{m['falhas_dice_abaixo_0.5']:5d}")
    print(f"\nsalvo em {saida.relative_to(path)}")


if __name__ == "__main__":
    main()
