"""Obtém Kvasir-SEG e HyperKvasir/cecum sem incluir os datasets no Git."""
import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from avaliacao_externa import ler_imagem, sha256
from config import ROOT
from download_negatives import baixar_zip, preparar as preparar_negativos

URL_KVASIR = "https://datasets.simula.no/downloads/kvasir-seg.zip"
QUANTIDADE = 1000


def validar_kvasir(pasta):
    imagens = sorted((pasta / "images").glob("*.jpg"))
    mascaras = sorted((pasta / "masks").glob("*.jpg"))
    if len(imagens) != QUANTIDADE or {p.name for p in imagens} != {p.name for p in mascaras}:
        raise ValueError(f"Esperados {QUANTIDADE} pares imagem/máscara em {pasta}")
    for imagem in imagens:
        img = ler_imagem(imagem)
        mask = ler_imagem(pasta / "masks" / imagem.name)
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Dimensões incompatíveis: {imagem.name}")


def preparar_kvasir(destino, tentativas=5, timeout=600):
    destino = Path(destino).resolve()
    pasta, arquivo = destino / "Kvasir-SEG", destino / "kvasir-seg.zip"
    if pasta.exists():
        validar_kvasir(pasta)
        print(f"{QUANTIDADE} pares Kvasir-SEG já disponíveis; download dispensado.")
        return
    baixar_zip(URL_KVASIR, arquivo, "Kvasir-SEG", tentativas, timeout)
    with zipfile.ZipFile(arquivo) as z:
        if z.testzip() is not None:
            raise ValueError(f"ZIP corrompido: {arquivo}")
        with tempfile.TemporaryDirectory(prefix=".kvasir-", dir=destino) as temp:
            staging = Path(temp)
            for nome in ("images", "masks"):
                (staging / nome).mkdir()
            for membro in z.infolist():
                relativo = PurePosixPath(membro.filename)
                if membro.is_dir() or relativo.suffix.lower() != ".jpg":
                    continue
                if len(relativo.parts) < 2 or relativo.parts[-2] not in {"images", "masks"}:
                    continue
                # Usa só pasta conhecida e basename; não extrai caminhos arbitrários.
                target = staging / relativo.parts[-2] / relativo.name
                with z.open(membro) as fonte, target.open("xb") as out:
                    shutil.copyfileobj(fonte, out)
            validar_kvasir(staging)
            (staging / "proveniencia.json").write_text(json.dumps({
                "fonte": "Kvasir-SEG", "url": URL_KVASIR, "pares": QUANTIDADE,
                "zip_sha256": sha256(arquivo), "data_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")
            shutil.move(str(staging), str(pasta))
    print(f"{QUANTIDADE} pares validados em {pasta}; ZIP preservado.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--tentativas", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if min(args.tentativas, args.timeout) < 1:
        parser.error("Tentativas e timeout devem ser positivos")
    try:
        preparar_kvasir(args.destino, args.tentativas, args.timeout)
        preparar_negativos(args.destino / "hyperkvasir", args.tentativas, args.timeout)
    except (OSError, ValueError, zipfile.BadZipFile) as erro:
        parser.exit(2, f"Erro: {erro}\n")


if __name__ == "__main__":
    main()
