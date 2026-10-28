"""Prepara os mesmos splits e máscaras para YOLO e U-Net, sem treinar modelos."""
import argparse
import hashlib
import json
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from evaluation_core import ROOT, ler_imagem, sha256

# configuracao do preparo dos dados, propria deste script
SEED = 42                                   # semente do split por grupo
CLASSES = ["polyp"]                         # uma classe: a tarefa aqui e detectar polipo
DADOS = ROOT / "data/processed/Kvasir-SEG"  # saida do preprocess

LIMIAR_SIMILARIDADE = 0.90


# algumas imagens de treino sao quase identicas, pois sao frames consecutivos. Entao caso caissem
# em splits diferentes, as imagens de teste seriam praticamente iguais as de treino, e o modelo teria um desempenho muito acima.
# Entao agrupamos imagens quase-identicas e mantemos cada grupo num unico split.
def agrupar_quase_duplicatas(files, limiar=LIMIAR_SIMILARIDADE, lado=128):
    if not files:
        return []
    # carregar as imagens em escala de cinza, redimensionar para lado x lado, normalizar e calcular a matriz de similaridade
    vetores = []
    for f in files:
        cinza = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if cinza is None:
            raise ValueError(f"Imagem ilegível: {f}")
        vetores.append(cv2.resize(cinza, (lado, lado)).astype(np.float32).ravel())
    t = np.stack(vetores)
    # normalizar cada vetor de caracteristicas para ter media 0 e norma 1, para que o produto interno seja a correlacao
    t -= t.mean(1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    correlacao = t @ t.T
    # Frames com bytes idênticos também ficam juntos, inclusive imagens constantes.
    # agrupar imagens com correlacao acima do limiar, usando union-find
    pai = list(range(len(files)))

    def find(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def unir(i, j):
        a, b = find(i), find(j)
        if a != b:
            pai[a] = b

    # para cada par de imagens com correlacao acima do limiar, unir os grupos
    for i, j in zip(*np.where(np.triu(correlacao > limiar, 1))):
        unir(i, j)
    hashes = {}
    for i, f in enumerate(files):
        digest = sha256(f)
        if digest in hashes:
            unir(i, hashes[digest])
        hashes[digest] = i
    # para cada grupo, criar uma lista de arquivos correspondentes
    grupos = {}
    for i, f in enumerate(files):
        grupos.setdefault(find(i), []).append(f)
    return list(grupos.values())


# realizar split de treino, validacao e teste, mantendo cada grupo de imagens quase-duplicadas no mesmo split
def split_por_grupo(files, rng, limiar=LIMIAR_SIMILARIDADE):
    grupos = agrupar_quase_duplicatas(files, limiar)
    # embaralhar os grupos para que a ordem dos arquivos nao influencie o split
    rng.shuffle(grupos)
    # cotas de cada split, em numero de imagens
    cotas = {"train": 0.8 * len(files), "val": 0.1 * len(files), "test": 0.1 * len(files)}
    out = {k: [] for k in cotas}
    # para cada grupo, alocar no split que tem mais espaco restante, para que os splits fiquem balanceados
    for grupo in sorted(grupos, key=len, reverse=True):
        alvo = max(cotas, key=lambda k: cotas[k] - len(out[k]))
        out[alvo].extend(grupo)
    print(f"{len(files)} imagens em {len(grupos)} grupos", flush=True)
    return out


# separar imagens em treino, teste e validacao
def preparar(raw, negativos, saida, seed=SEED, limiar=LIMIAR_SIMILARIDADE, sem_negativos=False,
             recriar=False, classes=("polyp",)):
    raw, negativos, saida = Path(raw).resolve(), Path(negativos).resolve(), Path(saida).resolve()
    positivos = sorted((raw / "images").glob("*.jpg"))
    fundos = [] if sem_negativos else sorted(negativos.glob("*.jpg"))
    if not positivos:
        raise ValueError(f"Kvasir-SEG ausente em {raw}")
    if not sem_negativos and not fundos:
        raise ValueError("Negativos ausentes: execute src/deteccao/download_dataset.py --apenas-negativos; --sem-negativos é um experimento diferente")
    if not 0 < limiar <= 1:
        raise ValueError("Limiar deve estar em (0,1]")
    nomes = [p.stem for p in positivos + fundos]
    if len(set(nomes)) != len(nomes):
        raise ValueError("Há nomes de imagens repetidos entre as fontes")
    # A remoção só é permitida dentro do destino, nunca numa fonte ou seu ancestral.
    if any(saida == fonte or saida in fonte.parents or fonte in saida.parents for fonte in (raw, negativos)):
        raise ValueError("A saída deve ser separada das fontes raw e negativos")
    if saida.exists() and not recriar:
        raise FileExistsError(f"{saida} já existe. Reutilize os dados ou use --recriar explicitamente.")
    for p in positivos:
        if not (raw / "masks" / p.name).is_file():
            raise FileNotFoundError(f"Máscara original ausente: {p.name}")
    rng = random.Random(seed)
    print("Agrupando positivos:", flush=True)
    splits_pos = split_por_grupo(positivos, rng, limiar)
    # Mantém a ordem do pipeline original: shuffle dos negativos antes do agrupamento.
    # os negativos tambem vem de video: mesmo agrupamento
    rng.shuffle(fundos)
    print("Agrupando negativos:", flush=True)
    splits_neg = split_por_grupo(fundos, rng, limiar)
    for s in splits_pos:
        if not splits_pos[s]:
            raise ValueError(f"Split {s} sem positivos; confira quantidade de grupos e limiar")
    saida.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preprocess-", dir=saida.parent))
    try:
        # to train the yolo model, the masks are in binary, so we need to convert the masks to txt containing the coordinates of the masks in yolo format
        # the masks are in jpg format and contains 3 channels
        # so we will binarize to have 2 classes: background and polyp
        # and preprocess them with morphological operations to remove noise and fill holes
        # saving them in png format to avoid compression artifacts from jpg
        # create the folders if they don't exist
        masks = staging / "masks_binary"
        masks.mkdir()
        for p in positivos:
            # carregar a mascara em escala de cinza
            img = ler_imagem(p)
            mask = cv2.imread(str(raw / "masks" / p.name), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != img.shape[:2]:
                raise ValueError(f"Máscara ilegível ou tamanho incompatível: {p.name}")
            # binarizar a mascara: 0 para fundo, 1 para polipo
            binary = (mask > 127).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            # abertura para remover ruido
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            #fechamento para preencher buracos
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            if not binary.any():
                raise ValueError(f"Máscara positiva vazia após processamento: {p.name}")
            # salvar a mascara binaria como PNG para evitar ruidos pela compressao jpeg
            if not cv2.imwrite(str(masks / f"{p.stem}.png"), binary):
                raise OSError(f"Falha ao salvar máscara: {p.name}")
        # converter as mascaras binarias para o formato YOLO
        from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
        labels = staging / "labels_all"
        labels.mkdir()
        convert_segment_masks_to_yolo_seg(masks_dir=str(masks), output_dir=str(labels), classes=1)
        manifest = {"seed": seed, "limiar_similaridade": limiar, "classes": list(classes),
                    "fontes": ["Kvasir-SEG"] + ([] if sem_negativos else ["HyperKvasir/cecum"]),
                    "fracoes_alvo": [0.8, 0.1, 0.1], "splits": {}}
        vistos = {}
        # para cada split, criar as pastas de imagens e labels, e copiar os arquivos correspondentes
        for split in splits_pos:
            imdir, lbdir = staging / "images" / split, staging / "labels" / split
            imdir.mkdir(parents=True)
            lbdir.mkdir(parents=True)
            registros = []
            for negativo, arquivos in ((False, splits_pos[split]), (True, splits_neg[split])):
                for p in arquivos:
                    digest = sha256(p)
                    if digest in vistos and vistos[digest] != split:
                        raise ValueError("Imagem idêntica em splits distintos; confira as fontes")
                    vistos[digest] = split
                    # estamos copiando a imagem para a pasta de imagens, e o label correspondente para a pasta de labels
                    shutil.copy2(p, imdir / p.name)
                    label = lbdir / f"{p.stem}.txt"
                    # imagens de fundo: o YOLO trata como negativo toda imagem cujo .txt esteja vazio.
                    # sem elas o modelo nunca ve mucosa normal e nao ha como medir falso positivo por frame.
                    if negativo:
                        label.write_text("", encoding="utf-8")   # .txt vazio = imagem de fundo
                    else:
                        fonte = labels / label.name
                        if not fonte.is_file() or not fonte.read_text().strip():
                            raise ValueError(f"Polígono YOLO ausente: {p.name}")
                        shutil.copy2(fonte, label)
                    registros.append({"imagem": f"images/{split}/{p.name}", "sha256": digest,
                                      "negativo": negativo,
                                      "mascara_sha256": None if negativo else sha256(masks / f"{p.stem}.png")})
            manifest["splits"][split] = registros
            print(f"{split}: {len(splits_pos[split])} positivos + {len(splits_neg[split])} negativos", flush=True)
        # Hash sem timestamp ou caminhos locais: identifica o mesmo conjunto em outra máquina.
        manifest["id_conjunto"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        manifest["data_utc"] = datetime.now(timezone.utc).isoformat()
        (staging / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # escrever o arquivo data.yaml para o treinamento do modelo YOLO.
        # os nomes vem do cenario, e nao fixos: um cenario de tipo tem varias classes
        nomes = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
        (staging / "data.yaml").write_text(
            f"path: {json.dumps(str(saida))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{nomes}\n",
            encoding="utf-8")
        if saida.exists():
            shutil.rmtree(saida)  # apenas com --recriar e depois de gerar/validar tudo
        staging.rename(saida)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/Kvasir-SEG")
    # imagens de ceco normal (HyperKvasir): frames de colon SEM polipo, usadas como fundo
    parser.add_argument("--negativos", type=Path, default=ROOT / "data/raw/hyperkvasir/cecum")
    parser.add_argument("--saida", type=Path, default=DADOS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limiar", type=float, default=LIMIAR_SIMILARIDADE)
    parser.add_argument("--sem-negativos", action="store_true")
    parser.add_argument("--recriar", action="store_true")
    args = parser.parse_args()
    try:
        manifest = preparar(args.raw, args.negativos, args.saida, args.seed,
                            args.limiar, args.sem_negativos, args.recriar, CLASSES)
    except (OSError, ValueError) as erro:
        parser.exit(2, f"Erro: {erro}\n")
    print("Conjunto:", manifest["id_conjunto"])
    print("Dados prontos em", args.saida.resolve())


if __name__ == "__main__":
    main()
