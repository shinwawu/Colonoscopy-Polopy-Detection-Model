"""
A visualizacao foi feita com IA 
modelo foi Gemini 3.5
Visualizacao da U-Net: desenha predicoes em imagens e plota as curvas do treino.

Reune o que a Ultralytics entrega pronto no lado do YOLO — o `yolo predict` e os
graficos que o train() salva sozinho. Do lado da U-Net cada um precisa ser escrito.

Cada predicao sai na resolucao ORIGINAL da imagem: a rede preve em 384x384 e a
mascara e reescalada de volta, para dar para inspecionar a lesao de perto.

  python visualize_unet.py                        # split de teste inteiro
  python visualize_unet.py --origem ../fotos      # uma pasta qualquer
  python visualize_unet.py --origem img.jpg       # uma imagem so
  python visualize_unet.py --painel               # imagem | probabilidade | overlay
  python visualize_unet.py --graficos             # curvas.png + predicoes.png
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_unet import (AREA_MINIMA, BASE, DESVIO, IMGSZ, KvasirSeg, MEDIA, NOME,
                        remove_blobs)
from unet import UNet

path = Path(__file__).resolve().parents[2]
data_path = path / "data/processed/Kvasir-SEG"
masks_path = data_path / "masks_binary"
run_path = path / "runs" / NOME

VERDE, VERMELHO = (0, 255, 0), (0, 0, 255)      # BGR: verde = anotacao, vermelho = predicao


def load_model(pesos):
    modelo = UNet(n_channels=3, n_classes=1, base=BASE).cuda().eval()
    modelo.load_state_dict(torch.load(pesos))
    return modelo


@torch.no_grad()
def predict(modelo, bgr):
    # mesma transformacao do KvasirSeg: resize para IMGSZ e normalizacao ImageNet
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0 - MEDIA) / DESVIO
    x = torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        logit = modelo(x)
    prob = torch.sigmoid(logit.float())[0, 0].cpu().numpy()

    pred = (prob > 0.5).astype(np.uint8)
    # filtro de area minima aplicado na grade da rede, onde AREA_MINIMA foi calibrado
    if AREA_MINIMA > 0 and pred.sum():
        n, rot, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < AREA_MINIMA:
                pred[rot == k] = 0

    # devolve tudo na resolucao original da imagem
    h, w = bgr.shape[:2]
    return (cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST),
            cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR))


def draw_overlay(bgr, pred, gt=None, alpha=0.35):
    # preenchimento translucido mostra a area; o contorno mostra a borda exata
    out = bgr.copy()
    pintura = out.copy()
    pintura[pred > 0] = VERMELHO
    out = cv2.addWeighted(pintura, alpha, out, 1 - alpha, 0)
    for mascara, cor in ((gt, VERDE), (pred, VERMELHO)):
        if mascara is None:
            continue
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contornos, -1, cor, 2)
    return out


def draw_text(img, linhas):
    # texto com contorno preto por baixo, para ficar legivel em mucosa clara
    for i, (texto, cor) in enumerate(linhas):
        p = (10, 24 + i * 22)
        cv2.putText(img, texto, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, texto, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 1, cv2.LINE_AA)


def heatmap(prob):
    # onde a rede hesitou: azul = certeza de fundo, vermelho = certeza de lesao
    return cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)


def curves():
    h = json.load(open(run_path / "historico.json"))
    ep = [e["epoca"] for e in h]
    melhor = max(range(len(h)), key=lambda i: h[i]["dice"])

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    paineis = [
        ("perda", [("perda_treino", "treino"), ("perda", "validacao")], "BCE + Dice"),
        ("Dice (validacao)", [("dice", "dice")], "Dice"),
        ("IoU (validacao)", [("iou", "iou")], "IoU"),
        ("falso positivo em fundos", [("fp_fundos", "fp")], "fracao"),
    ]
    for a, (titulo, series, ylabel) in zip(ax.ravel(), paineis):
        for chave, rotulo in series:
            a.plot(ep, [e[chave] for e in h], label=rotulo, lw=1.5)
        # linha vertical na epoca do checkpoint escolhido
        a.axvline(h[melhor]["epoca"], color="gray", ls="--", lw=1,
                  label=f"melhor (ep {h[melhor]['epoca']})")
        a.set_title(titulo); a.set_xlabel("epoca"); a.set_ylabel(ylabel)
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_path / "curvas.png", dpi=110)
    print(f"curvas.png  -> melhor dice {h[melhor]['dice']:.4f} na epoca {h[melhor]['epoca']}")


def predictions(n_por_faixa=4):
    modelo = UNet(n_channels=3, n_classes=1, base=BASE).cuda().eval()
    modelo.load_state_dict(torch.load(run_path / "weights/best.pt"))
    ds = KvasirSeg("test")

    # roda o teste guardando o Dice de cada imagem positiva, junto do indice dela
    itens = []
    with torch.no_grad():
        for b, (x, y) in enumerate(DataLoader(ds, batch_size=8, num_workers=4)):
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                pred = (torch.sigmoid(modelo(x).float()) > 0.5).float()
            for k, (p_i, y_i) in enumerate(zip(pred, y)):
                if y_i.sum() == 0:
                    continue
                p_i = remove_blobs(p_i, AREA_MINIMA)
                inter = (p_i * y_i).sum().item()
                soma = (p_i.sum() + y_i.sum()).item()
                itens.append((2 * inter / soma if soma else 0.0, b * 8 + k,
                              p_i[0].cpu().numpy().astype(np.uint8)))
    itens.sort(key=lambda t: t[0])
    meio = len(itens) // 2
    faixas = [("piores", itens[:n_por_faixa]),
              ("medianas", itens[meio - n_por_faixa // 2: meio - n_por_faixa // 2 + n_por_faixa]),
              ("melhores", itens[-n_por_faixa:][::-1])]

    fig, ax = plt.subplots(3, n_por_faixa, figsize=(3.2 * n_por_faixa, 10))
    for linha, (nome, grupo) in enumerate(faixas):
        for col, (dice, idx, pred) in enumerate(grupo):
            a = ax[linha, col]
            img = cv2.cvtColor(cv2.imread(str(ds.imgs[idx])), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMGSZ, IMGSZ))
            gt = cv2.imread(str(ds.masks / f"{ds.imgs[idx].stem}.png"), cv2.IMREAD_GRAYSCALE)
            gt = (cv2.resize(gt, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
            # verde = ground truth, vermelho = predicao do modelo
            for mascara, cor in ((gt, (0, 255, 0)), (pred, (255, 0, 0))):
                contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contornos, -1, cor, 2)
            a.imshow(img); a.axis("off")
            a.set_title(f"{ds.imgs[idx].stem[:12]}\ndice {dice:.3f}", fontsize=9)
        # rotulo da faixa na lateral: set_ylabel nao aparece com axis("off")
        ax[linha, 0].text(-0.06, 0.5, nome, transform=ax[linha, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=12)
    fig.suptitle("verde = ground truth    vermelho = predicao", fontsize=11)
    fig.tight_layout()
    fig.savefig(run_path / "predicoes.png", dpi=110)
    d = np.array([t[0] for t in itens])
    print(f"predicoes.png -> piores {d[:3].round(3)}  mediana {np.median(d):.3f}  "
          f"melhores {d[-3:].round(3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default=str(data_path / "images/test"))
    ap.add_argument("--saida", default=str(run_path / "predicoes"))
    ap.add_argument("--pesos", default=str(run_path / "weights/best.pt"))
    ap.add_argument("--painel", action="store_true", help="imagem | probabilidade | overlay")
    ap.add_argument("--limite", type=int, default=0, help="processa so as N primeiras")
    ap.add_argument("--graficos", action="store_true",
                    help="gera curvas.png e predicoes.png em vez de desenhar imagens")
    a = ap.parse_args()

    if a.graficos:
        curves()
        predictions()
        print(f"\nsalvos em runs/{NOME}/")
        return

    origem = Path(a.origem)
    arquivos = sorted([p for p in origem.rglob("*") if p.suffix.lower() in {".jpg", ".png"}]) \
        if origem.is_dir() else [origem]
    if a.limite:
        arquivos = arquivos[:a.limite]
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)

    modelo = load_model(a.pesos)
    dices, sem_gt, fp = [], 0, 0
    for f in arquivos:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        pred, prob = predict(modelo, bgr)

        mp = masks_path / f"{f.stem}.png"
        gt = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) if mp.exists() else None

        linhas = [(f"area prevista: {int(pred.sum())} px", (255, 255, 255))]
        if gt is None:
            # imagem de fundo: nao ha lesao anotada, entao qualquer pixel previsto e erro
            sem_gt += 1
            fp += int(pred.sum() > 0)
            linhas.append(("sem anotacao (fundo)" + ("  FALSO POSITIVO" if pred.sum() else "  ok"),
                           VERMELHO if pred.sum() else VERDE))
        else:
            inter = int((pred & gt).sum())
            soma = int(pred.sum() + gt.sum())
            d = 2 * inter / soma if soma else 0.0
            dices.append(d)
            linhas.append((f"dice {d:.3f}", VERDE if d >= 0.8 else VERMELHO))

        out = draw_overlay(bgr, pred, gt)
        draw_text(out, linhas)
        if a.painel:
            calor = heatmap(prob)
            bruto = bgr.copy()
            draw_text(bruto, [(f.stem[:20], (255, 255, 255))])
            out = np.hstack([bruto, calor, out])
        cv2.imwrite(str(saida / f"{f.stem}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # relative_to so vale se a saida estiver dentro do projeto
    rel = saida.relative_to(path) if saida.is_relative_to(path) else saida
    print(f"{len(arquivos)} imagens -> {rel}")
    if dices:
        d = np.array(dices)
        print(f"  positivos: {len(d)}  dice medio {d.mean():.4f}  mediana {np.median(d):.4f}")
    if sem_gt:
        print(f"  fundos: {sem_gt}  falsos positivos {fp} ({fp / sem_gt:.1%})")


if __name__ == "__main__":
    main()
