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

from avaliacao_externa import ler_imagem, sha256
from config import DADOS, ROOT

LIMIAR_SIMILARIDADE = 0.90


def agrupar_quase_duplicatas(files, limiar=LIMIAR_SIMILARIDADE, lado=128):
    if not files:
        return []
    vetores = []
    for f in files:
        cinza = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if cinza is None:
            raise ValueError(f"Imagem ilegível: {f}")
        vetores.append(cv2.resize(cinza, (lado, lado)).astype(np.float32).ravel())
    t = np.stack(vetores)
    t -= t.mean(1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    correlacao = t @ t.T
    # Frames com bytes idênticos também ficam juntos, inclusive imagens constantes.
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

    for i, j in zip(*np.where(np.triu(correlacao > limiar, 1))):
        unir(i, j)
    hashes = {}
    for i, f in enumerate(files):
        digest = sha256(f)
        if digest in hashes:
            unir(i, hashes[digest])
        hashes[digest] = i
    grupos = {}
    for i, f in enumerate(files):
        grupos.setdefault(find(i), []).append(f)
    return list(grupos.values())


def split_por_grupo(files, rng, limiar=LIMIAR_SIMILARIDADE):
    grupos = agrupar_quase_duplicatas(files, limiar)
    rng.shuffle(grupos)
    cotas = {"train": 0.8 * len(files), "val": 0.1 * len(files), "test": 0.1 * len(files)}
    out = {k: [] for k in cotas}
    for grupo in sorted(grupos, key=len, reverse=True):
        alvo = max(cotas, key=lambda k: cotas[k] - len(out[k]))
        out[alvo].extend(grupo)
    print(f"{len(files)} imagens em {len(grupos)} grupos", flush=True)
    return out


def preparar(raw, negativos, saida, seed=42, limiar=LIMIAR_SIMILARIDADE, sem_negativos=False, recriar=False):
    raw, negativos, saida = Path(raw).resolve(), Path(negativos).resolve(), Path(saida).resolve()
    positivos = sorted((raw / "images").glob("*.jpg"))
    fundos = [] if sem_negativos else sorted(negativos.glob("*.jpg"))
    if not positivos:
        raise ValueError(f"Kvasir-SEG ausente em {raw}")
    if not sem_negativos and not fundos:
        raise ValueError("Negativos ausentes: execute src/download_negatives.py; --sem-negativos é um experimento diferente")
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
    rng.shuffle(fundos)
    print("Agrupando negativos:", flush=True)
    splits_neg = split_por_grupo(fundos, rng, limiar)
    for s in splits_pos:
        if not splits_pos[s]:
            raise ValueError(f"Split {s} sem positivos; confira quantidade de grupos e limiar")
    saida.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preprocess-", dir=saida.parent))
    try:
        masks = staging / "masks_binary"
        masks.mkdir()
        for p in positivos:
            img = ler_imagem(p)
            mask = cv2.imread(str(raw / "masks" / p.name), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != img.shape[:2]:
                raise ValueError(f"Máscara ilegível ou tamanho incompatível: {p.name}")
            binary = (mask > 127).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            if not binary.any():
                raise ValueError(f"Máscara positiva vazia após processamento: {p.name}")
            if not cv2.imwrite(str(masks / f"{p.stem}.png"), binary):
                raise OSError(f"Falha ao salvar máscara: {p.name}")
        from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
        labels = staging / "labels_all"
        labels.mkdir()
        convert_segment_masks_to_yolo_seg(masks_dir=str(masks), output_dir=str(labels), classes=1)
        manifest = {"seed": seed, "limiar_similaridade": limiar,
                    "fontes": ["Kvasir-SEG"] + ([] if sem_negativos else ["HyperKvasir/cecum"]),
                    "fracoes_alvo": [0.8, 0.1, 0.1], "splits": {}}
        vistos = {}
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
                    shutil.copy2(p, imdir / p.name)
                    label = lbdir / f"{p.stem}.txt"
                    if negativo:
                        label.write_text("", encoding="utf-8")
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
        (staging / "data.yaml").write_text(
            f"path: {json.dumps(str(saida))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: polyp\n",
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
    parser.add_argument("--negativos", type=Path, default=ROOT / "data/raw/hyperkvasir/cecum")
    parser.add_argument("--saida", type=Path, default=DADOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limiar", type=float, default=LIMIAR_SIMILARIDADE)
    parser.add_argument("--sem-negativos", action="store_true")
    parser.add_argument("--recriar", action="store_true")
    args = parser.parse_args()
    try:
        manifest = preparar(args.raw, args.negativos, args.saida, args.seed, args.limiar,
                            args.sem_negativos, args.recriar)
    except (OSError, ValueError) as erro:
        parser.exit(2, f"Erro: {erro}\n")
    print("Conjunto:", manifest["id_conjunto"])
    print("Dados prontos em", args.saida.resolve())


if __name__ == "__main__":
    main()
