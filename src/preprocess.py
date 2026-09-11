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

# separar imagens em treino, teste e validacao
files = sorted((data_path / "images").glob("*.jpg"))
random.shuffle(files)
n = len(files)
bounds = {
    "train": (0, int(0.8 * n)),
    "val": (int(0.8 * n), int(0.9 * n)),
    "test": (int(0.9 * n), n),}

for split, (lo, hi) in bounds.items():
    # para cada split, criar as pastas de imagens e labels, e copiar os arquivos correspondentes
    images_dir = yolo_data_path / "images" / split
    labels_dir = yolo_data_path / "labels" / split
    shutil.rmtree(images_dir, ignore_errors=True)
    shutil.rmtree(labels_dir, ignore_errors=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path in files[lo:hi]:
        # estamos copiando a imagem para a pasta de imagens, e o label correspondente para a pasta de labels
        shutil.copy2(img_path, images_dir / img_path.name)
        shutil.copy2(labels_all_path / f"{img_path.stem}.txt", labels_dir / f"{img_path.stem}.txt")
    print(f"{split}: {hi - lo} imagens")

# escrever o arquivo data.yaml para o treinamento do modelo YOLO
(yolo_data_path / "data.yaml").write_text(
    f"path: {yolo_data_path.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: polyp\n"
)

