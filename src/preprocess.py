import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

random.seed(42)
np.random.seed(42)

# raw images
path = Path(__file__).parent.parent
data_path = path / "data/raw/Kvasir-SEG"
yolo_data_path = path / "data/processed/Kvasir-SEG"
binary_masks_path = yolo_data_path / "masks_binary"
labels_all_path = yolo_data_path / "labels_all"
# imagens de ceco normal (HyperKvasir): frames de colon SEM polipo, usadas como fundo
negativos_path = path / "data/raw/hyperkvasir/cecum"
# fracao dos negativos disponiveis a usar (1.0 = todos os 1009, ~1:1 com os positivos)
FRACAO_NEGATIVOS = 1.0

# to train the yolo model, the masks are in binary, so we need to convert the masks to txt containing the coordinates of the masks in yolo format
# the masks are in jpg format and contains 3 channels
# so we will binarize to have 2 classes: background and polyp
# and preprocess them with morphological operations to remove noise and fill holes
# saving them in png format to avoid compression artifacts from jpg
# 


# create the folders if they don't exist
yolo_data_path.mkdir(parents=True, exist_ok=True)
labels_all_path.mkdir(parents=True, exist_ok=True)
binary_masks_path.mkdir(parents=True, exist_ok=True)


for mask_path in sorted(data_path.glob("masks/*.jpg")):
    # carregar a mascara em escala de cinza
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    # binarizar a mascara: 0 para fundo, 1 para polipo
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    # abertura para remover ruido
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    #fechamento para preencher buracos
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # salvar a mascara binaria como PNG para evitar ruidos pela compressao jpeg
    if not cv2.imwrite(str(binary_masks_path / f"{mask_path.stem}.png"), binary_mask):
        raise IOError(f"falha ao escrever a mascara binaria: {mask_path.stem}.png")
# converter as mascaras binarias para o formato YOLO
convert_segment_masks_to_yolo_seg(
    masks_dir=str(binary_masks_path), output_dir=str(labels_all_path), classes=1
)

# algumas imagens de treino sao quase identicas, pois sao frames consecutivos. Entao caso caissem
# em splits diferentes, as imagens de teste seriam praticamente iguais as de treino, e o modelo teria um desempenho muito acima.
# Entao agrupamos imagens quase-identicas e mantemos cada grupo num unico split.
LIMIAR_SIMILARIDADE = 0.90


def agrupar_quase_duplicatas(files, limiar=LIMIAR_SIMILARIDADE, lado=128):
    # carregar as imagens em escala de cinza, redimensionar para lado x lado, normalizar e calcular a matriz de similaridade
    T = np.stack([
        cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (lado, lado)).astype(np.float32).ravel()
        for f in files
    ])
    # normalizar cada vetor de caracteristicas para ter media 0 e norma 1, para que o produto interno seja a correlacao
    T -= T.mean(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    C = T @ T.T

    # agrupar imagens com correlacao acima do limiar, usando union-find
    n = len(files)
    pai = list(range(n))

    def find(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    # para cada par de imagens com correlacao acima do limiar, unir os grupos
    for i, j in zip(*np.where(np.triu(C > limiar, 1))):
        a, b = find(i), find(j)
        if a != b:
            pai[a] = b
    grupos = {}
    # para cada grupo, criar uma lista de arquivos correspondentes
    for i in range(n):
        grupos.setdefault(find(i), []).append(files[i])
    return list(grupos.values())


def split_por_grupo(files, fracoes=(0.8, 0.1, 0.1)):
    # realizar split de treino, validacao e teste, mantendo cada grupo de imagens quase-duplicadas no mesmo split
    grupos = agrupar_quase_duplicatas(files)
    # embaralhar os grupos para que a ordem dos arquivos nao influencie o split
    random.shuffle(grupos)
    n = len(files)
    # cotas de cada split, em numero de imagens
    cotas = {"train": fracoes[0] * n, "val": fracoes[1] * n, "test": fracoes[2] * n}
    out = {k: [] for k in cotas}
    # para cada grupo, alocar no split que tem mais espaco restante, para que os splits fiquem balanceados
    for g in sorted(grupos, key=len, reverse=True):
        alvo = max(cotas, key=lambda k: cotas[k] - len(out[k]))
        out[alvo].extend(g)
    # imprimir quantos grupos foram formados, e quantos deles tem mais de uma imagem
    n_multi = sum(1 for g in grupos if len(g) > 1)
    print(f"  {len(grupos)} grupos ({n_multi} com >1 imagem) para {n} imagens")
    return out


# separar imagens em treino, teste e validacao
files = sorted((data_path / "images").glob("*.jpg"))
print("agrupando positivos:")
splits_pos = split_por_grupo(files)

for split, arquivos in splits_pos.items():
    # para cada split, criar as pastas de imagens e labels, e copiar os arquivos correspondentes
    images_dir = yolo_data_path / "images" / split
    labels_dir = yolo_data_path / "labels" / split
    shutil.rmtree(images_dir, ignore_errors=True)
    shutil.rmtree(labels_dir, ignore_errors=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path in arquivos:
        # estamos copiando a imagem para a pasta de imagens, e o label correspondente para a pasta de labels
        shutil.copy2(img_path, images_dir / img_path.name)
        shutil.copy2(labels_all_path / f"{img_path.stem}.txt", labels_dir / f"{img_path.stem}.txt")
    print(f"{split}: {len(arquivos)} imagens")

# imagens de fundo: o YOLO trata como negativo toda imagem cujo .txt esteja vazio.
# sem elas o modelo nunca ve mucosa normal e nao ha como medir falso positivo por frame.
if negativos_path.exists():
    negativos = sorted(negativos_path.glob("*.jpg"))
    random.shuffle(negativos)
    negativos = negativos[: int(len(negativos) * FRACAO_NEGATIVOS)]
    # os negativos tambem vem de video: mesmo agrupamento
    print("agrupando negativos:")
    splits_neg = split_por_grupo(negativos)
    for split, arquivos in splits_neg.items():
        images_dir = yolo_data_path / "images" / split
        labels_dir = yolo_data_path / "labels" / split
        for img_path in arquivos:
            shutil.copy2(img_path, images_dir / img_path.name)
            # .txt vazio = imagem de fundo
            (labels_dir / f"{img_path.stem}.txt").write_text("")
        print(f"{split}: +{len(arquivos)} negativos")
else:
    print(f"AVISO: {negativos_path} nao existe, treinando sem imagens de fundo")

# escrever o arquivo data.yaml para o treinamento do modelo YOLO
(yolo_data_path / "data.yaml").write_text(
    f"path: {yolo_data_path.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: polyp\n"
)

