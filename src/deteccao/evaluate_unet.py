# step de validacao e teste do modelo U-Net
import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_unet import AREA_MINIMA, BASE, IMGSZ, KvasirSeg, NOME, SEED, remove_blobs
from unet import UNet

path = Path(__file__).resolve().parents[2]
runs_path = path / "runs"
artifacts_path = path / "artifacts"

LIMIARES = np.round(np.arange(0.05, 0.96, 0.05), 2)


@torch.no_grad()
def probabilities(modelo, split):
    # roda a rede uma vez so e guarda os mapas de probabilidade, para poder varrer
    # os limiares sem repetir inferencia
    mapas, alvos = [], []
    for x, y in DataLoader(KvasirSeg(split), batch_size=8, num_workers=4, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.float16):
            logit = modelo(x.cuda(non_blocking=True))
        mapas.append(torch.sigmoid(logit.float()).cpu())
        alvos.append(y)
    return torch.cat(mapas), torch.cat(alvos)


def measure(probs, alvos, limiar, area_minima=AREA_MINIMA):
    dices, ious, fp, n_fundo = [], [], 0, 0
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    # decisao por imagem: tem ou nao tem polipo (para a matriz de confusao)
    verdadeiro, previsto = [], []
    for p_i, y_i in zip(probs, alvos):
        m = remove_blobs((p_i > limiar).float(), area_minima)
        tem_gt, tem_pred = y_i.sum() > 0, m.sum() > 0
        verdadeiro.append(int(tem_gt)); previsto.append(int(tem_pred))
        if not tem_gt:
            n_fundo += 1
            fp += int(tem_pred)
            continue
        inter = (m * y_i).sum().item()
        soma, uniao = (m.sum() + y_i.sum()).item(), ((m + y_i) > 0).sum().item()
        dices.append(2 * inter / soma if soma else 0.0)
        ious.append(inter / uniao if uniao else 0.0)
        tp_tot += inter
        fp_tot += (m * (1 - y_i)).sum().item()
        fn_tot += ((1 - m) * y_i).sum().item()
    d = np.array(dices)
    return {
        "limiar": float(limiar),
        "dice": float(d.mean()), "dice_mediana": float(np.median(d)),
        "dice_erro_padrao": float(d.std(ddof=1) / np.sqrt(len(d))),
        "iou": float(np.mean(ious)),
        # precisao e recall agregados em pixels, para a curva PR
        "precisao_px": tp_tot / (tp_tot + fp_tot) if tp_tot + fp_tot else 0.0,
        "recall_px": tp_tot / (tp_tot + fn_tot) if tp_tot + fn_tot else 0.0,
        "fp_fundos": fp / n_fundo if n_fundo else 0.0,
        "falhas_dice_abaixo_0.5": int((d < 0.5).sum()),
        "acertos_dice_acima_0.9": int((d > 0.9).sum()),
        "n_positivos": len(d), "n_fundos": n_fundo,
        "_dices": d, "_verdadeiro": np.array(verdadeiro), "_previsto": np.array(previsto),
    }


def confusion_matrix(m, saida):
    v, p = m["_verdadeiro"], m["_previsto"]
    # mesma orientacao do YOLO: Predicted na vertical, True na horizontal
    cm = np.array([[int(((p == 1) & (v == 1)).sum()), int(((p == 1) & (v == 0)).sum())],
                   [int(((p == 0) & (v == 1)).sum()), int(((p == 0) & (v == 0)).sum())]])
    rotulos = ["polipo", "fundo"]
    for normalizada in (False, True):
        dados = cm / cm.sum(axis=0, keepdims=True) if normalizada else cm
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.imshow(dados, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{dados[i, j]:.2f}" if normalizada else f"{cm[i, j]}",
                        ha="center", va="center",
                        color="white" if dados[i, j] > dados.max() / 2 else "black")
        ax.set_xticks([0, 1], rotulos); ax.set_yticks([0, 1], rotulos)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title("Matriz de confusao" + (" normalizada" if normalizada else "")
                     + "\n(por imagem: tem ou nao tem polipo)", fontsize=10)
        fig.tight_layout()
        fig.savefig(saida / ("confusion_matrix_normalized.png" if normalizada
                             else "confusion_matrix.png"), dpi=110)
        plt.close(fig)
    return cm.tolist()


def plots(varredura_val, varredura_teste, melhor, m_teste, saida):
    lim = [v["limiar"] for v in varredura_teste]
    pr = [v["precisao_px"] for v in varredura_teste]
    rc = [v["recall_px"] for v in varredura_teste]
    f1 = [2 * p * r / (p + r) if p + r else 0 for p, r in zip(pr, rc)]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(rc, pr, "o-", ms=3)
    ax[0, 0].set_xlabel("recall"); ax[0, 0].set_ylabel("precisao")
    ax[0, 0].set_title("curva PR (pixel, teste)")

    for serie, rotulo in ((pr, "precisao"), (rc, "recall"), (f1, "F1")):
        ax[0, 1].plot(lim, serie, "o-", ms=3, label=rotulo)
    ax[0, 1].set_xlabel("limiar"); ax[0, 1].set_title("precisao / recall / F1 por limiar")
    ax[0, 1].legend(fontsize=8)

    # a escolha do limiar acontece aqui, e so com a curva de validacao
    ax[1, 0].plot([v["limiar"] for v in varredura_val],
                  [v["dice"] for v in varredura_val], "o-", ms=3, label="validacao")
    ax[1, 0].plot(lim, [v["dice"] for v in varredura_teste], "o-", ms=3,
                  alpha=0.5, label="teste (so conferencia)")
    ax[1, 0].axvline(melhor, color="gray", ls="--", lw=1, label=f"escolhido: {melhor:.2f}")
    ax[1, 0].set_xlabel("limiar"); ax[1, 0].set_ylabel("Dice")
    ax[1, 0].set_title("Dice por limiar"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].hist(m_teste["_dices"], bins=20, range=(0, 1), edgecolor="white")
    ax[1, 1].axvline(m_teste["dice"], color="red", ls="--", lw=1.2,
                     label=f"media {m_teste['dice']:.3f}")
    ax[1, 1].axvline(m_teste["dice_mediana"], color="green", ls="--", lw=1.2,
                     label=f"mediana {m_teste['dice_mediana']:.3f}")
    ax[1, 1].set_xlabel("Dice por imagem"); ax[1, 1].set_ylabel("imagens")
    ax[1, 1].set_title("distribuicao do Dice (teste)"); ax[1, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(saida / "curvas_teste.png", dpi=110)
    plt.close(fig)


def batches(modelo, limiar, saida, n_lotes=3):
    # equivalente aos val_batchN_labels.jpg / _pred.jpg do YOLO.
    # a ordem alfabetica do split poe todos os negativos na frente (eles tem nome
    # UUID e os positivos comecam com "cju"), entao um lote sequencial sairia sem
    # nenhum polipo. embaralha com semente fixa para misturar, mantendo reprodutivel.
    ds = KvasirSeg("test")
    ds.imgs = sorted(ds.imgs)
    random.Random(SEED).shuffle(ds.imgs)
    dl = DataLoader(ds, batch_size=8, num_workers=4)
    for b, (x, y) in enumerate(dl):
        if b >= n_lotes:
            break
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            prob = torch.sigmoid(modelo(x.cuda()).float()).cpu()
        for nome, mascaras in (("labels", y), ("pred", prob)):
            fig, ax = plt.subplots(2, 4, figsize=(13, 7))
            for k, a in enumerate(ax.ravel()):
                if k >= len(x):
                    a.axis("off"); continue
                idx = b * 8 + k
                img = cv2.cvtColor(cv2.imread(str(ds.imgs[idx])), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMGSZ, IMGSZ))
                m = mascaras[k][0].numpy()
                m = remove_blobs(torch.from_numpy((m > limiar).astype(np.float32))[None],
                                 AREA_MINIMA)[0].numpy().astype(np.uint8) \
                    if nome == "pred" else (m > 0.5).astype(np.uint8)
                cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cont, -1, (255, 0, 0) if nome == "pred" else (0, 255, 0), 2)
                a.imshow(img); a.axis("off")
                a.set_title(ds.imgs[idx].stem[:14], fontsize=7)
            fig.suptitle(f"lote {b} — {'predicao' if nome == 'pred' else 'anotacao'}")
            fig.tight_layout()
            fig.savefig(saida / f"test_batch{b}_{nome}.jpg", dpi=100)
            plt.close(fig)


def main():
    checkpoint = torch.load(runs_path / NOME / "weights/best.pt", weights_only=True)
    # o train_unet.py grava {state_dict, config, epoca}; o baseline original gravava o state_dict puro
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    base = int(checkpoint.get("config", {}).get("base", BASE))
    modelo = UNet(n_channels=3, n_classes=1, base=base).cuda().eval()
    modelo.load_state_dict(state)

    saida = runs_path / f"{NOME}_teste"
    saida.mkdir(parents=True, exist_ok=True)

    # 1. varredura na VALIDACAO -> escolhe o limiar
    pv, av = probabilities(modelo, "val")
    varredura_val = [measure(pv, av, t) for t in LIMIARES]
    melhor = max(varredura_val, key=lambda v: v["dice"])["limiar"]
    print(f"limiar escolhido na validacao: {melhor:.2f} "
          f"(dice val {max(v['dice'] for v in varredura_val):.4f}, "
          f"contra {[v for v in varredura_val if v['limiar'] == 0.5][0]['dice']:.4f} em 0.50)")

    # 2. TESTE com o limiar escolhido
    pt, at = probabilities(modelo, "test")
    varredura_teste = [measure(pt, at, t) for t in LIMIARES]
    m = measure(pt, at, melhor)

    # 3. saidas
    cm = confusion_matrix(m, saida)
    plots(varredura_val, varredura_teste, melhor, m, saida)
    batches(modelo, melhor, saida)

    resultado = {
        "nome": f"{NOME}_teste",
        "data": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "modelo": f"unet (base={BASE})", "imgsz": IMGSZ, "split": "test",
            "limiar": melhor, "limiar_escolhido_em": "val", "area_minima": AREA_MINIMA,
            "pesos": f"runs/{NOME}/weights/best.pt",
        },
        "metricas": {k: v for k, v in m.items() if not k.startswith("_")},
        "matriz_confusao": {"ordem": "[[pred_polipo/true_polipo, pred_polipo/true_fundo], "
                                     "[pred_fundo/true_polipo, pred_fundo/true_fundo]]",
                            "valores": cm},
        "varredura_val": [{k: v[k] for k in ("limiar", "dice", "fp_fundos")}
                          for v in varredura_val],
    }
    (saida / "metricas.json").write_text(json.dumps(resultado, indent=2))
    artifacts_path.mkdir(parents=True, exist_ok=True)
    (artifacts_path / f"{NOME}_teste.json").write_text(json.dumps(resultado, indent=2))

    print(f"\ndice {m['dice']:.4f}  mediana {m['dice_mediana']:.4f}  iou {m['iou']:.4f}  "
          f"fp_fundos {m['fp_fundos']:.4f}")
    print(f"matriz: {cm}")
    print(f"saida -> runs/{NOME}_teste/")


if __name__ == "__main__":
    main()
