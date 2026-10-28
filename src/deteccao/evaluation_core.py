"""Leitura, métricas e saídas comuns aos experimentos, mais ROOT e seleção de device.

Aqui fica só o que não é configuração de modelo: cada script define os próprios
hiperparâmetros, caminhos de dados e nomes de execução.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# src/deteccao/evaluation_core.py -> sobe tres niveis ate a raiz do repositorio
ROOT = Path(__file__).resolve().parents[2]

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def dispositivo(valor="auto"):
    import torch

    if valor == "auto":
        valor = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(valor).isdigit():
        valor = f"cuda:{valor}"
    device = torch.device(valor)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponível neste ambiente; confira a instalação ou use --device cpu")
    return device


def sha256(arquivo):
    with Path(arquivo).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ler_imagem(arquivo):
    imagem = cv2.imread(str(arquivo))
    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {arquivo}")
    return imagem


def carregar_casos(origem, anotacoes=None):
    """CSV opcional: imagem,mascara,presenca,origem.

    Imagens são relativas à pasta de entrada; máscaras, à pasta do CSV.
    Presença aceita 1 (pólipo), 0 (negativo confirmado) ou vazio (desconhecido).
    Uma máscara informada também determina a presença. Sem anotação não há GT.
    """
    origem = Path(origem).resolve()
    if origem.is_file() and origem.suffix.lower() in EXTENSOES:
        arquivos, raiz = [origem], origem.parent
    elif origem.is_dir():
        arquivos = sorted(p for p in origem.rglob("*") if p.suffix.lower() in EXTENSOES and p.is_file())
        raiz = origem
    else:
        raise FileNotFoundError(f"Imagem ou pasta não encontrada: {origem}")
    if not arquivos:
        raise ValueError(f"Nenhuma imagem encontrada em {origem}")

    registros = {}
    if anotacoes is not None:
        anotacoes = Path(anotacoes).resolve()
        with anotacoes.open(encoding="utf-8-sig", newline="") as f:
            leitor = csv.DictReader(f)
            if not {"imagem", "mascara", "presenca", "origem"}.issubset(leitor.fieldnames or []):
                raise ValueError("O CSV precisa das colunas imagem,mascara,presenca,origem")
            for linha in leitor:
                nome = linha["imagem"].strip()
                relativo = Path(nome)
                if not nome or relativo.is_absolute() or ".." in relativo.parts:
                    raise ValueError(f"Caminho de imagem inválido no CSV: {nome!r}")
                nome = relativo.as_posix()
                if nome in registros:
                    raise ValueError(f"Imagem repetida no CSV: {nome}")
                if not (raiz / nome).is_file():
                    raise FileNotFoundError(f"Imagem anotada não encontrada: {raiz / nome}")
                presenca = linha["presenca"].strip()
                if presenca not in {"", "0", "1"}:
                    raise ValueError(f"Presença deve ser 0, 1 ou vazia: {nome}")
                mascara = linha["mascara"].strip()
                mp = (anotacoes.parent / mascara).resolve() if mascara else None
                if mp is not None and not mp.is_file():
                    raise FileNotFoundError(f"Máscara anotada não encontrada: {mp}")
                registros[nome] = {
                    "mascara": str(mp) if mp else None,
                    "presenca": int(presenca) if presenca else None,
                    "origem": linha["origem"].strip(),
                }

    casos = []
    for arquivo in arquivos:
        nome = arquivo.relative_to(raiz).as_posix()
        registro = registros.get(nome, {"mascara": None, "presenca": None, "origem": ""})
        casos.append({"imagem": nome, "caminho": str(arquivo), **registro})
    return casos


def carregar_referencia(caso, formato):
    """Não converte uma máscara ausente em negativo, salvo presença=0 explícita."""
    presenca = caso["presenca"]
    gt = None
    if caso["mascara"]:
        gt = cv2.imread(caso["mascara"], cv2.IMREAD_GRAYSCALE)
        if gt is None or gt.shape != formato:
            raise ValueError(f"Máscara ilegível ou com dimensões diferentes da imagem: {caso['imagem']}")
        gt = (gt > 0).astype(np.uint8)
        observado = int(gt.any())
        if presenca is not None and presenca != observado:
            raise ValueError(f"Máscara e presença discordam: {caso['imagem']}")
        presenca = observado
    elif presenca == 0:
        gt = np.zeros(formato, np.uint8)
    return gt, presenca


def carregar_casos_internos(dados, split="test"):
    """Lê o mesmo split para os dois modelos; só label vazio confirma negativo."""
    if split not in {"train", "val", "test"}:
        raise ValueError("Split deve ser train, val ou test")
    dados = Path(dados).resolve()
    casos = carregar_casos(dados / "images" / split)
    for caso in casos:
        stem = Path(caso["imagem"]).stem
        label = dados / "labels" / split / f"{stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Label ausente: {label}. Execute o preprocessamento.")
        positivo = bool(label.read_text(encoding="utf-8").strip())
        mascara = dados / "masks_binary" / f"{stem}.png"
        if positivo and not mascara.is_file():
            raise FileNotFoundError(f"Máscara de um positivo ausente: {mascara}")
        caso.update(mascara=str(mascara) if positivo else None,
                    presenca=int(positivo), origem=f"interno/{split}/" + ("Kvasir-SEG" if positivo else "HyperKvasir-cecum"))
    return casos


def filtrar_mascara(mascara, area_minima=500):
    mascara = (mascara > 0).astype(np.uint8)
    if area_minima < 0:
        raise ValueError("A área mínima não pode ser negativa")
    if area_minima and mascara.any():
        n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < area_minima:
                mascara[rotulos == k] = 0
    return mascara


def avaliar_mascara(pred, gt, presenca):
    detectou = bool(pred.any())
    dice = iou = None
    if gt is not None and gt.any():
        inter = int(np.count_nonzero((pred > 0) & (gt > 0)))
        soma = int(np.count_nonzero(pred)) + int(np.count_nonzero(gt))
        uniao = int(np.count_nonzero((pred > 0) | (gt > 0)))
        dice, iou = 2 * inter / soma, inter / uniao
    resultado = "sem_anotacao"
    if presenca is not None:
        resultado = ("TP" if detectou else "FN") if presenca else ("FP" if detectou else "TN")
    return {"presenca": presenca, "detectou": int(detectou), "resultado": resultado,
            "dice": dice, "iou": iou}


def resumir(linhas):
    """Resume a detecção por imagem e a segmentação dos positivos com máscara.

    Os nomes do relatório original permanecem como aliases. Sem máscaras não
    existe média/mediana; o erro padrão amostral exige pelo menos duas máscaras.
    Positivos apenas com presença anotada entram na sensibilidade, mas não no
    Dice/IoU. Negativos e imagens sem anotação também não entram nessas métricas.
    """
    contagens = {k: sum(linha["resultado"] == k for linha in linhas)
                 for k in ("TP", "FN", "FP", "TN", "sem_anotacao")}
    n_pos = contagens["TP"] + contagens["FN"]
    n_neg = contagens["FP"] + contagens["TN"]
    dice = [l["dice"] for l in linhas if l["dice"] is not None]
    iou = [l["iou"] for l in linhas if l["iou"] is not None]
    dice_medio = float(np.mean(dice)) if dice else None
    iou_medio = float(np.mean(iou)) if iou else None
    taxa_fp = contagens["FP"] / n_neg if n_neg else None
    return {"n_imagens": len(linhas), "n_positivos": n_pos, "n_negativos": n_neg,
            "n_sem_anotacao": contagens["sem_anotacao"], "n_mascaras_positivas": len(dice),
            "dice_medio": dice_medio,
            "dice_mediana": float(np.median(dice)) if dice else None,
            "dice_erro_padrao": float(np.std(dice, ddof=1) / np.sqrt(len(dice))) if len(dice) > 1 else None,
            "falhas_dice_abaixo_0.5": sum(d < 0.5 for d in dice),
            "acertos_dice_acima_0.9": sum(d > 0.9 for d in dice),
            "iou_medio": iou_medio,
            "sensibilidade_por_imagem": contagens["TP"] / n_pos if n_pos else None,
            "taxa_fp_por_imagem": taxa_fp,
            "dice": dice_medio, "iou": iou_medio,
            "fp_fundos": taxa_fp, "n_fundos": n_neg,
            **contagens}


def sobrepor(imagem, mascara):
    saida = imagem.copy()
    pixels = mascara > 0
    saida[pixels] = (0.55 * saida[pixels] + 0.45 * np.array([0, 170, 255])).astype(np.uint8)
    return saida


def salvar_imagem(caminho, imagem):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(caminho), imagem):
        raise OSError(f"Falha ao salvar {caminho}")


def executar_avaliacao(casos, predizer, saida, config, grade=384, area_minima=500):
    """predizer(BGR) retorna (máscara binária, detalhes do modelo).

    A máscara deve ter a resolução original, salvo se config declarar
    formato_predicao=[altura, largura], como na grade nativa da U-Net.
    Avaliação e filtro usam uma única projeção à grade comum dos modelos;
    só as saídas visuais voltam à resolução original. Uma pasta existente
    nunca é sobrescrita.
    """
    if not casos or grade <= 0 or area_minima < 0:
        raise ValueError("Informe imagens, grade positiva e área mínima não negativa")
    saida = Path(saida).resolve()
    saida.mkdir(parents=True, exist_ok=False)
    linhas, entradas = [], []
    for caso in casos:
        imagem = ler_imagem(caso["caminho"])
        h, w = imagem.shape[:2]
        gt, presenca = carregar_referencia(caso, (h, w))
        pred, detalhes = predizer(imagem)
        formato_esperado = tuple(config.get("formato_predicao", (h, w)))
        if pred.shape != formato_esperado:
            raise ValueError(f"Predição com dimensões erradas: {caso['imagem']}")
        pred = cv2.resize(pred.astype(np.uint8), (grade, grade), interpolation=cv2.INTER_NEAREST)
        pred = filtrar_mascara(pred, area_minima)
        if gt is not None:
            gt = cv2.resize(gt, (grade, grade), interpolation=cv2.INTER_NEAREST)
        # Mantém a presença da anotação original mesmo se uma lesão muito pequena
        # desaparecer no redimensionamento. Dice/IoU ficam indisponíveis nesse caso.
        metricas = avaliar_mascara(pred, gt, presenca)
        original = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        nome = Path(caso["imagem"] + ".png")  # a.jpg e a.png não colidem
        mp = Path("mascaras") / nome
        op = Path("sobreposicoes") / nome
        salvar_imagem(saida / mp, original * 255)
        salvar_imagem(saida / op, sobrepor(imagem, original))
        linhas.append({"imagem": caso["imagem"], "origem": caso["origem"],
                       **metricas, "pixels_previstos_grade": int(pred.sum()),
                       "mascara_predita": mp.as_posix(), "sobreposicao": op.as_posix(),
                       **detalhes})
        entradas.append({**caso, "sha256": sha256(caso["caminho"]),
                         "mascara_sha256": sha256(caso["mascara"]) if caso["mascara"] else None})

    resultado = {"data_utc": datetime.now(timezone.utc).isoformat(),
                 "config": {**config, "grade_avaliacao": grade, "area_minima_grade": area_minima},
                 "entradas": entradas, "resumo": resumir(linhas), "por_imagem": linhas}
    with (saida / "por_imagem.csv").open("w", encoding="utf-8", newline="") as f:
        campos = list(dict.fromkeys(k for linha in linhas for k in linha))
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)
    (saida / "resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return resultado
