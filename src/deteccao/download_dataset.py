"""Obtém Kvasir-SEG e HyperKvasir/cecum sem incluir os datasets no Git.

Os dois downloads ficam aqui: o Kvasir-SEG traz os positivos com máscara, e o
cecum do HyperKvasir traz a mucosa normal usada como negativo.
"""
import argparse
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from evaluation_core import ROOT, ler_imagem, sha256

URL_KVASIR = "https://datasets.simula.no/downloads/kvasir-seg.zip"
PARES_KVASIR = 1000

URL_CECUM = "https://files.osf.io/v1/resources/mh9sj/providers/osfstorage/5dfb5b530236b8000c746a8e/?zip="
IMAGENS_CECUM = 1009


def baixar_zip(url, arquivo, nome, tentativas=5, timeout=600):
    """Mantém ZIP completo em cache; substitui o .part só depois de validar CRC."""
    arquivo = Path(arquivo)
    if arquivo.exists():
        return
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    parcial = arquivo.with_suffix(arquivo.suffix + ".part")
    for tentativa in range(1, tentativas + 1):
        try:
            print(f"Baixando {nome} (tentativa {tentativa}/{tentativas})...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "polyp-segmentation-project/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resposta, parcial.open("wb") as f:
                shutil.copyfileobj(resposta, f, length=1024 * 1024)
            with zipfile.ZipFile(parcial) as z:
                if z.testzip() is not None:
                    raise ValueError("ZIP com CRC inválido")
            parcial.replace(arquivo)
            return
        except (OSError, ValueError, zipfile.BadZipFile):
            if tentativa == tentativas:
                raise
            print("Download interrompido; tentando novamente em 5 segundos.", flush=True)
            time.sleep(5)


# ---------------------------------------------------------------- Kvasir-SEG

def validar_kvasir(pasta):
    imagens = sorted((pasta / "images").glob("*.jpg"))
    mascaras = sorted((pasta / "masks").glob("*.jpg"))
    if len(imagens) != PARES_KVASIR or {p.name for p in imagens} != {p.name for p in mascaras}:
        raise ValueError(f"Esperados {PARES_KVASIR} pares imagem/máscara em {pasta}")
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
        print(f"{PARES_KVASIR} pares Kvasir-SEG já disponíveis; download dispensado.")
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
                "fonte": "Kvasir-SEG", "url": URL_KVASIR, "pares": PARES_KVASIR,
                "zip_sha256": sha256(arquivo), "data_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")
            shutil.move(str(staging), str(pasta))
    print(f"{PARES_KVASIR} pares validados em {pasta}; ZIP preservado.")


# -------------------------------------------------- HyperKvasir/cecum (negativos)

def validar_cecum(pasta, quantidade=IMAGENS_CECUM):
    imagens = sorted(pasta.glob("*.jpg"))
    if len(imagens) != quantidade:
        raise ValueError(f"Esperadas {quantidade} imagens em {pasta}; encontradas {len(imagens)}")
    for p in imagens:
        ler_imagem(p)
    return imagens


def preparar_cecum(destino, tentativas=5, timeout=600):
    destino = Path(destino).resolve()
    pasta, arquivo = destino / "cecum", destino / "cecum.zip"
    if pasta.exists():
        validar_cecum(pasta)
        print(f"{IMAGENS_CECUM} imagens já disponíveis em {pasta}; download dispensado.")
        return
    baixar_zip(URL_CECUM, arquivo, "HyperKvasir/cecum", tentativas, timeout)
    with zipfile.ZipFile(arquivo) as z:
        if z.testzip() is not None:
            raise ValueError(f"ZIP corrompido: {arquivo}; preserve/renomeie o arquivo antes de baixar novamente")
        with tempfile.TemporaryDirectory(prefix=".cecum-", dir=destino) as temp:
            staging = Path(temp)
            membros = [m for m in z.infolist() if not m.is_dir()]
            if len(membros) != IMAGENS_CECUM:
                raise ValueError(f"Conteúdo do ZIP diferente das {IMAGENS_CECUM} imagens esperadas")
            nomes = set()
            for membro in membros:
                nome = Path(membro.filename).name
                if Path(nome).suffix.lower() != ".jpg" or nome in nomes:
                    raise ValueError(f"Membro inesperado ou repetido no ZIP: {membro.filename}")
                nomes.add(nome)
                # Copia pelo nome simples; não extrai caminhos arbitrários do ZIP.
                with z.open(membro) as fonte, (staging / nome).open("wb") as out:
                    shutil.copyfileobj(fonte, out)
            validar_cecum(staging)
            shutil.move(str(staging), str(pasta))
    (destino / "proveniencia.json").write_text(json.dumps({
        "fonte": "HyperKvasir", "subconjunto": "labeled-images/lower-gi-tract/anatomical-landmarks/cecum",
        "url": URL_CECUM, "quantidade": IMAGENS_CECUM, "zip_sha256": sha256(arquivo),
        "data_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"{IMAGENS_CECUM} imagens validadas em {pasta}. ZIP preservado para reutilização.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--tentativas", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600, help="Timeout de leitura em segundos")
    parser.add_argument("--apenas-negativos", action="store_true",
                        help="baixa só o cecum do HyperKvasir, pulando o Kvasir-SEG")
    args = parser.parse_args()
    if min(args.tentativas, args.timeout) < 1:
        parser.error("Tentativas e timeout devem ser positivos")
    try:
        if not args.apenas_negativos:
            preparar_kvasir(args.destino, args.tentativas, args.timeout)
        preparar_cecum(args.destino / "hyperkvasir", args.tentativas, args.timeout)
    except (OSError, ValueError, zipfile.BadZipFile) as erro:
        parser.exit(2, f"Erro: {erro}\n")


if __name__ == "__main__":
    main()
