import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from unet import UNet

path = Path(__file__).parent.parent
data_path = path / "data/processed/Kvasir-SEG"
runs_path = path / "runs"
artifacts_path = path / "artifacts"

NOME = "unet-baseline"
EPOCAS, IMGSZ, BATCH = 100, 384, 8
LEARNING_RATE, EARLY_STOP, SEED = 1e-3, 25, 42
# tamanho da primeira camada convolucional, que dobra a cada downsampling
BASE = 32         
# area minima (em pixels) de um componente conexo na predicao. um polipo real ocupa
# ~22k px em 384x384, entao mancha de poucas centenas de pixels e ruido, nao lesao.
AREA_MINIMA = 500

# normalizacao da imagem, usando media e desvio padrao do ImageNet
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# data augmentation: flip horizontal, vertical e rotacao de ate 15 graus
def aumenta(img, msk, rng):
    if rng.random() < 0.5:                      # girar a imagem para a esquerda/direita (flip horizontal)
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.5:                      # girar a imagem para cima/baixo (flip vertical)
        img, msk = img[::-1], msk[::-1]
    ang = rng.uniform(-15, 15)                  # rotacionar a imagem em ate 15 graus
    h, w = msk.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR)
    #aqui usamos INTER_NEAREST para nao criar pixels intermediarios na mascara, que e binaria
    msk = cv2.warpAffine(np.ascontiguousarray(msk), M, (w, h), flags=cv2.INTER_NEAREST)
    return img, msk


class KvasirSeg(Dataset):
    def __init__(self, split, treino=False):
        # split e "train", "val" ou "test"; treino=True ativa data augmentation
        self.imgs = sorted((data_path / "images" / split).glob("*.jpg"))
        self.masks = data_path / "masks_binary"
        self.treino = treino

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        # ler a imagem e a mascara correspondente; se nao houver mascara, e um negativo (fundo)
        f = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)

        mp = self.masks / f"{f.stem}.png"
        if mp.exists():
            msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        else:
            # se nao houver mascara, criamos uma mascara vazia
            msk = np.zeros(img.shape[:2], np.uint8)

        # redimensiona o par de imagem+mascara para o tamanho fixo da rede
        img = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)

        # realiza data augmentation apenas no treino
        if self.treino:
            img, msk = aumenta(img, msk, random.Random(SEED + i + random.randint(0, 10**6)))

        # normaliza a imagem para [0,1], subtrai a media e divide pelo desvio padrao
        img = (np.ascontiguousarray(img).astype(np.float32) / 255.0 - MEDIA) / DESVIO
        return (torch.from_numpy(img).permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(msk).astype(np.float32))[None])


def loss_dice(logit, alvo, eps=1.0):
    # calculando o dice score, pois devido a desbalanceamento de classes
    p = torch.sigmoid(logit)
    num = 2 * (p * alvo).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + alvo.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()

# loss function combinando BCE e Dice, para lidar com desbalanceamento de classes
bce = nn.BCEWithLogitsLoss()
def loss_bce_dice(logit, alvo):
    return 0.5 * bce(logit, alvo) + 0.5 * loss_dice(logit, alvo)

# pos-processamento: apaga da predicao os componentes conexos menores que area_minima.
# nao muda o modelo, so limpa a saida dele; area_minima=0 desliga o filtro.
def remove_blobs(p_i, area_minima=AREA_MINIMA):
    if area_minima <= 0 or p_i.sum() == 0:
        return p_i
    m = p_i[0].cpu().numpy().astype(np.uint8)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # o rotulo 0 e o fundo, por isso o range comeca em 1
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < area_minima:
            m[rotulos == k] = 0
    return torch.from_numpy(m).to(p_i.device, torch.float32)[None]


# aqui comparamos o desempenho do modelo em imagens de validação ou teste, calculando métricas como Dice, IoU e taxa de falso positivo
@torch.no_grad()
def avalia(modelo, loader, limiar=0.5, area_minima=AREA_MINIMA):
    # aqui estamos avaliando o modelo em um conjunto de dados (loader) e calculando métricas como Dice, IoU e taxa de falso positivo
    modelo.eval()
    # inicializando listas e contadores para armazenar métricas
    dices, ious, fp, n_fundo, perdas = [], [], 0, 0, []
    for x, y in loader:
        # movendo os dados para a GPU e habilitando o autocast para reduzir o uso de memória
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            # fazendo a inferência do modelo e calculando a perda
            logit = modelo(x)
        perdas.append(loss_bce_dice(logit.float(), y).item())
        pred = (torch.sigmoid(logit.float()) > limiar).float()
        # calculando métricas para cada imagem no batch
        for p_i, y_i in zip(pred, y):
            p_i = remove_blobs(p_i, area_minima)
            inter = (p_i * y_i).sum().item()
            soma, uniao = (p_i.sum() + y_i.sum()).item(), ((p_i + y_i) > 0).sum().item()
            # se a imagem nao tiver nenhum pixel positivo (ou seja, e um fundo), contamos como falso positivo se o modelo previu algum pixel positivo
            if y_i.sum() == 0:                  
                n_fundo += 1
                fp += int(p_i.sum() > 0)
            else:
                dices.append(2 * inter / soma if soma else 0.0)
                ious.append(inter / uniao if uniao else 0.0)
    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "fp_fundos": fp / n_fundo if n_fundo else 0.0,
        "perda": float(np.mean(perdas)),
        "n_positivos": len(dices),
        "n_fundos": n_fundo,
    }


def main():
    # criando dataset de treino, validacao e teste, com data augmentation apenas no treino
    tr = DataLoader(KvasirSeg("train", treino=True), batch_size=BATCH, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    va = DataLoader(KvasirSeg("val"), batch_size=BATCH, num_workers=4, pin_memory=True)
    te = DataLoader(KvasirSeg("test"), batch_size=BATCH, num_workers=4, pin_memory=True)
    print(f"train={len(tr.dataset)}  val={len(va.dataset)}  test={len(te.dataset)}")
    # init do modelo
    modelo = UNet(n_channels=3, n_classes=1, base=BASE).cuda()
    print(f"parametros: {sum(p.numel() for p in modelo.parameters()) / 1e6:.1f}M")

    # optimizer AdamW com weight decay para regularizacao, e learning rate inicial definido
    opt = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # scheduler para regular o learning rate ao longo do treinamento, usando Cosine Annealing
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS)
    # smp or scaler para reduzir o uso de memória durante o treinamento, usando mixed precision training (AMP)
    scaler = torch.amp.GradScaler("cuda")

    # criando diretório para salvar os pesos do modelo e inicializando variáveis para acompanhar o melhor desempenho
    pesos_dir = runs_path / NOME / "weights"
    pesos_dir.mkdir(parents=True, exist_ok=True)
    melhor, melhor_ep, historico = -1.0, 0, []

    # train process, para cada epoca, o modelo é treinado em batches de imagens e máscaras, calculando a perda e atualizando os pesos do modelo. 
    # após cada época, o modelo é avaliado no conjunto de validação e as métricas são registradas. 
    # Se o desempenho melhorar, os pesos do modelo são salvos. O treinamento é interrompido se não houver melhora após um número definido de épocas (early stopping).
    for ep in range(1, EPOCAS + 1):
        modelo.train()
        soma = 0.0
        for x, y in tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                p = loss_bce_dice(modelo(x), y)
            scaler.scale(p).backward()
            scaler.step(opt)
            scaler.update()
            soma += p.item()
        sched.step()

        m = avalia(modelo, va)
        historico.append({"epoca": ep, "perda_treino": soma / len(tr), **m})
        print(f"ep {ep:3d}/{EPOCAS}  treino={soma/len(tr):.4f}  val_dice={m['dice']:.4f} "
              f"iou={m['iou']:.4f}  fp={m['fp_fundos']:.3f}")

        # o checkpoint e escolhido pelo Dice de VALIDACAO, nunca pelo teste
        if m["dice"] > melhor:
            melhor, melhor_ep = m["dice"], ep
            torch.save(modelo.state_dict(), pesos_dir / "best.pt")
        elif ep - melhor_ep >= EARLY_STOP:
            print(f"early stopping: {EARLY_STOP} epocas sem melhora")
            break

    # avaliacao final no TESTE, com o melhor checkpoint
    modelo.load_state_dict(torch.load(pesos_dir / "best.pt"))
    teste = avalia(modelo, te)

    resultado = {
        "nome": NOME,
        "data": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "modelo": f"unet (base={BASE})",
            "parametros_M": round(sum(p.numel() for p in modelo.parameters()) / 1e6, 2),
            "epocas": len(historico), "imgsz": IMGSZ, "batch": BATCH, "lr": LEARNING_RATE,
            "melhor_epoca": melhor_ep, "split": "test",
            "pesos": f"runs/{NOME}/weights/best.pt",
        },
        "metricas": teste,
        "val_melhor_dice": melhor,
    }
    artifacts_path.mkdir(parents=True, exist_ok=True)
    (artifacts_path / f"{NOME}.json").write_text(json.dumps(resultado, indent=2))
    (runs_path / NOME / "historico.json").write_text(json.dumps(historico, indent=2))
    print(json.dumps(resultado["metricas"], indent=2))


if __name__ == "__main__":
    main()
