"""Treina a U-Net e avalia o melhor checkpoint de validação no teste interno."""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from evaluation_core import carregar_casos_internos, carregar_referencia, filtrar_mascara, ler_imagem, sha256
from evaluation_core import ROOT, dispositivo
from unet import UNet

# Defaults do baseline original; novos experimentos têm um diretório próprio.
# ---- configuracao deste modelo, toda neste arquivo ----
runs_path, artifacts_path = ROOT / "runs", ROOT / "artifacts"
data_path = ROOT / "data/processed/Kvasir-SEG"
NOME = "unet-deteccao"
SEED = 42
EPOCAS, IMGSZ, BATCH = 100, 384, 8
# BASE e o tamanho da primeira camada convolucional, que dobra a cada downsampling.
# AREA_MINIMA e a area minima (em pixels) de um componente conexo na predicao: um polipo
# real ocupa ~22k px em 384x384, entao mancha de poucas centenas de pixels e ruido, nao lesao.
LEARNING_RATE, EARLY_STOP, BASE, AREA_MINIMA = 1e-3, 25, 32, 500

# normalizacao da imagem, usando media e desvio padrao do ImageNet.
# vao gravados no checkpoint, entao a inferencia nao depende deste arquivo
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)


# data augmentation: flip horizontal, vertical e rotacao de ate 15 graus
def aumenta(img, msk, rng):
    if rng.random() < 0.5:                      # girar a imagem para a esquerda/direita (flip horizontal)
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.5:                      # girar a imagem para cima/baixo (flip vertical)
        img, msk = img[::-1], msk[::-1]
    h, w = msk.shape
    matriz = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), matriz, (w, h), flags=cv2.INTER_LINEAR)
    #aqui usamos INTER_NEAREST para nao criar pixels intermediarios na mascara, que e binaria
    msk = cv2.warpAffine(np.ascontiguousarray(msk), matriz, (w, h), flags=cv2.INTER_NEAREST)
    return img, msk


# cada worker do DataLoader e um processo novo: aqui ele recebe uma semente derivada da
# semente base do torch, para numpy e random nao comecarem iguais nos quatro processos
def semeia_worker(worker_id):
    semente = torch.initial_seed() % 2**31
    np.random.seed(semente)
    random.seed(semente)


class KvasirSeg(Dataset):
    def __init__(self, split, treino=False, dados=None, imgsz=IMGSZ, seed=SEED):
        # split e "train", "val" ou "test"; treino=True ativa data augmentation
        self.casos = carregar_casos_internos(dados or data_path, split)
        self.imgs = [Path(c["caminho"]) for c in self.casos]
        self.treino, self.imgsz, self.seed = treino, imgsz, seed
        # a epoca entra na semente da augmentation: o treino chama define_epoca() a cada
        # volta, entao o sorteio muda de epoca para epoca sem deixar de ser reprodutivel
        self.epoca = 0

    def define_epoca(self, epoca):
        self.epoca = epoca

    def __len__(self):
        return len(self.casos)

    def __getitem__(self, i):
        caso = self.casos[i]
        # ler a imagem e a mascara correspondente; se nao houver mascara, e um negativo (fundo)
        img = ler_imagem(caso["caminho"])
        msk, _ = carregar_referencia(caso, img.shape[:2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # redimensiona o par de imagem+mascara para o tamanho fixo da rede
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        # realiza data augmentation apenas no treino
        if self.treino:
            # so (seed, epoca, indice) decidem o sorteio; usar o random global aqui
            # tornava a augmentation irreprodutivel e o --seed, decorativo
            rng = random.Random(self.seed * 1_000_003 + self.epoca * 10_007 + i)
            img, msk = aumenta(img, msk, rng)
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


bce = nn.BCEWithLogitsLoss()


# loss function combinando BCE e Dice, para lidar com desbalanceamento de classes
def loss_bce_dice(logit, alvo):
    return 0.5 * bce(logit, alvo) + 0.5 * loss_dice(logit, alvo)


# pos-processamento: apaga da predicao os componentes conexos menores que area_minima.
# nao muda o modelo, so limpa a saida dele; area_minima=0 desliga o filtro.
def remove_blobs(p_i, area_minima=AREA_MINIMA):
    m = filtrar_mascara(p_i[0].cpu().numpy(), area_minima)
    return torch.from_numpy(m).to(p_i.device, torch.float32)[None]


# aqui comparamos o desempenho do modelo em imagens de validação ou teste, calculando métricas como Dice, IoU e taxa de falso positivo
@torch.no_grad()
def avalia(modelo, loader, limiar=0.5, area_minima=AREA_MINIMA, amp=True):
    # aqui estamos avaliando o modelo em um conjunto de dados (loader) e calculando métricas como Dice, IoU e taxa de falso positivo
    device = next(modelo.parameters()).device
    modelo.eval()
    # inicializando listas e contadores para armazenar métricas
    dices, ious, fp, n_fundo, perdas = [], [], 0, 0, []
    for x, y in loader:
        # movendo os dados para a GPU e habilitando o autocast para reduzir o uso de memória
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            # fazendo a inferência do modelo e calculando a perda
            logit = modelo(x)
        perdas.append(loss_bce_dice(logit.float(), y).item())
        pred = (torch.sigmoid(logit.float()) > limiar).float()
        # calculando métricas para cada imagem no batch
        for p_i, y_i in zip(pred, y):
            p_i = remove_blobs(p_i, area_minima)
            inter = (p_i * y_i).sum().item()
            soma = (p_i.sum() + y_i.sum()).item()
            uniao = ((p_i + y_i) > 0).sum().item()
            # se a imagem nao tiver nenhum pixel positivo (ou seja, e um fundo), contamos como falso positivo se o modelo previu algum pixel positivo
            if y_i.sum() == 0:
                n_fundo += 1
                fp += int(p_i.sum() > 0)
            else:
                dices.append(2 * inter / soma if soma else 0.0)
                ious.append(inter / uniao if uniao else 0.0)
    return {"dice": float(np.mean(dices)) if dices else None,
            "iou": float(np.mean(ious)) if ious else None,
            "fp_fundos": fp / n_fundo if n_fundo else None,
            "perda": float(np.mean(perdas)), "n_positivos": len(dices), "n_fundos": n_fundo}


def salvar_json(arquivo, valor):
    temporario = arquivo.with_suffix(".tmp")
    temporario.write_text(json.dumps(valor, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporario.replace(arquivo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=data_path)
    parser.add_argument("--runs", type=Path, default=runs_path,
                        help="Pasta de saída dos treinos e checkpoints")
    parser.add_argument("--artifacts", type=Path, default=artifacts_path,
                        help="Pasta de saída dos relatórios JSON")
    parser.add_argument("--nome", default=NOME)
    parser.add_argument("--epocas", type=int, default=EPOCAS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--base", type=int, default=BASE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=EARLY_STOP)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0 ou cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--area-minima", type=int, default=AREA_MINIMA)
    args = parser.parse_args(argv)
    if min(args.epocas, args.batch, args.base, args.patience) < 1 or args.lr <= 0:
        parser.error("Épocas, batch, base, patience e lr devem ser positivos")
    if args.imgsz < 32 or args.imgsz % 16 or args.workers < 0 or args.area_minima < 0:
        parser.error("imgsz deve ser múltiplo de 16 e >=32; workers e área devem ser não negativos")
    if Path(args.nome).name != args.nome or args.nome in {"", ".", ".."}:
        parser.error("--nome deve ser um nome de pasta simples")
    run = args.runs / args.nome
    artefato = args.artifacts / f"{args.nome}.json"
    if run.exists() or artefato.exists():
        parser.error("Esse experimento já existe; escolha outro --nome para preservar os resultados")
    try:
        device = dispositivo(args.device)
        # criando dataset de treino, validacao e teste, com data augmentation apenas no treino
        datasets = {s: KvasirSeg(s, treino=s == "train", dados=args.dados, imgsz=args.imgsz,
                               seed=args.seed)
                    for s in ("train", "val", "test")}
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # sem isso o backward das convolucoes escolhe algoritmos nao deterministicos e dois
    # treinos com a mesma seed divergem desde a primeira epoca
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # generator proprio: a ordem do shuffle deixa de depender de quem consumiu o RNG global antes
    gerador = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(d, batch_size=args.batch, shuffle=s == "train",
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            drop_last=s == "train" and len(d) >= args.batch,
                            generator=gerador if s == "train" else None,
                            worker_init_fn=semeia_worker)
               for s, d in datasets.items()}
    print("  ".join(f"{s}={len(d)}" for s, d in datasets.items()), flush=True)
    # init do modelo
    modelo = UNet(n_channels=3, n_classes=1, base=args.base).to(device)
    # optimizer AdamW com weight decay para regularizacao, e learning rate inicial definido
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler para regular o learning rate ao longo do treinamento, usando Cosine Annealing
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epocas)
    # smp or scaler para reduzir o uso de memória durante o treinamento, usando mixed precision training (AMP)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    # criando diretório para salvar os pesos do modelo e inicializando variáveis para acompanhar o melhor desempenho
    pesos_dir = run / "weights"
    pesos_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.dados / "split_manifest.json"
    config = {"modelo": "unet", "base": args.base, "imgsz": args.imgsz, "batch": args.batch,
              "epocas_solicitadas": args.epocas, "lr": args.lr, "seed": args.seed,
              "patience": args.patience, "workers": args.workers,
              "drop_last": {s: loader.drop_last for s, loader in loaders.items()},
              "device": str(device), "amp": scaler.is_enabled(), "limiar": 0.5,
              "area_minima": args.area_minima, "media": MEDIA.tolist(), "desvio": DESVIO.tolist(),
              "torch": str(torch.__version__), "dados": str(args.dados.resolve()),
              "split_manifest_sha256": sha256(manifest) if manifest.is_file() else None}
    salvar_json(run / "config.json", config)
    melhor, melhor_ep, historico = -1.0, 0, []

    # train process, para cada epoca, o modelo é treinado em batches de imagens e máscaras, calculando a perda e atualizando os pesos do modelo. 
    # após cada época, o modelo é avaliado no conjunto de validação e as métricas são registradas. 
    # Se o desempenho melhorar, os pesos do modelo são salvos. O treinamento é interrompido se não houver melhora após um número definido de épocas (early stopping).
    for ep in range(1, args.epocas + 1):
        datasets["train"].define_epoca(ep)
        modelo.train()
        soma = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                logit = modelo(x)
                perda = loss_bce_dice(logit.float(), y)
            scaler.scale(perda).backward()
            scaler.step(opt)
            scaler.update()
            soma += perda.item()
        sched.step()
        m = avalia(modelo, loaders["val"], area_minima=args.area_minima, amp=args.amp)
        if m["dice"] is None:
            raise ValueError("Validação sem máscara positiva; confira o split dos dados")
        historico.append({"epoca": ep, "perda_treino": soma / len(loaders["train"]), **m})
        print(f"ep {ep:3d}/{args.epocas} treino={historico[-1]['perda_treino']:.4f} "
              f"val_dice={m['dice']:.4f} iou={m['iou']:.4f} fp={m['fp_fundos']}", flush=True)
        # o checkpoint e escolhido pelo Dice de VALIDACAO, nunca pelo teste.
        # so o melhor vai para o disco: o last.pt nao era usado por nenhum script daqui
        if m["dice"] > melhor:
            melhor, melhor_ep = m["dice"], ep
            # Metadados viajam junto dos pesos; inferência não depende dos defaults do código.
            checkpoint = {"state_dict": modelo.state_dict(), "config": config, "epoca": ep}
            # escreve em .tmp e renomeia: um best.pt pela metade, se a maquina cair, nao
            # sobrescreve o ultimo bom
            temp = pesos_dir / "best.tmp"
            torch.save(checkpoint, temp)
            temp.replace(pesos_dir / "best.pt")
        salvar_json(run / "historico.json", historico)
        if ep - melhor_ep >= args.patience:
            break
    # avaliacao final no TESTE, com o melhor checkpoint
    best = pesos_dir / "best.pt"
    modelo.load_state_dict(torch.load(best, map_location=device, weights_only=True)["state_dict"])
    teste = avalia(modelo, loaders["test"], area_minima=args.area_minima, amp=args.amp)
    resultado = {"nome": args.nome, "data": datetime.now().isoformat(timespec="seconds"),
                 "config": {**config, "epocas": len(historico), "melhor_epoca": melhor_ep,
                            "split": "test", "pesos": str(best.resolve())},
                 "metricas": teste, "val_melhor_dice": melhor}
    args.artifacts.mkdir(parents=True, exist_ok=True)
    salvar_json(artefato, resultado)
    print(json.dumps(resultado, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
