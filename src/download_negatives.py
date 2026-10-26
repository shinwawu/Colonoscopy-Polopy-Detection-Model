"""Baixa/reutiliza o subconjunto cecum (1009 imagens) do HyperKvasir."""
import argparse
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from avaliacao_externa import ler_imagem, sha256
from config import ROOT

URL = "https://files.osf.io/v1/resources/mh9sj/providers/osfstorage/5dfb5b530236b8000c746a8e/?zip="
QUANTIDADE = 1009


def validar(pasta, quantidade=QUANTIDADE):
    imagens = sorted(pasta.glob("*.jpg"))
    if len(imagens) != quantidade:
        raise ValueError(f"Esperadas {quantidade} imagens em {pasta}; encontradas {len(imagens)}")
    for p in imagens:
        ler_imagem(p)
    return imagens


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


def preparar(destino, tentativas=5, timeout=600):
    destino = Path(destino).resolve()
    pasta, arquivo = destino / "cecum", destino / "cecum.zip"
    if pasta.exists():
        validar(pasta)
        print(f"{QUANTIDADE} imagens já disponíveis em {pasta}; download dispensado.")
        return
    baixar_zip(URL, arquivo, "HyperKvasir/cecum", tentativas, timeout)
    with zipfile.ZipFile(arquivo) as z:
        if z.testzip() is not None:
            raise ValueError(f"ZIP corrompido: {arquivo}; preserve/renomeie o arquivo antes de baixar novamente")
        with tempfile.TemporaryDirectory(prefix=".cecum-", dir=destino) as temp:
            staging = Path(temp)
            membros = [m for m in z.infolist() if not m.is_dir()]
            if len(membros) != QUANTIDADE:
                raise ValueError("Conteúdo do ZIP diferente das 1009 imagens esperadas")
            nomes = set()
            for membro in membros:
                nome = Path(membro.filename).name
                if Path(nome).suffix.lower() != ".jpg" or nome in nomes:
                    raise ValueError(f"Membro inesperado ou repetido no ZIP: {membro.filename}")
                nomes.add(nome)
                # Copia pelo nome simples; não extrai caminhos arbitrários do ZIP.
                with z.open(membro) as fonte, (staging / nome).open("wb") as out:
                    shutil.copyfileobj(fonte, out)
            validar(staging)
            shutil.move(str(staging), str(pasta))
    (destino / "proveniencia.json").write_text(json.dumps({
        "fonte": "HyperKvasir", "subconjunto": "labeled-images/lower-gi-tract/anatomical-landmarks/cecum",
        "url": URL, "quantidade": QUANTIDADE, "zip_sha256": sha256(arquivo),
        "data_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"{QUANTIDADE} imagens validadas em {pasta}. ZIP preservado para reutilização.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=ROOT / "data/raw/hyperkvasir")
    parser.add_argument("--tentativas", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600, help="Timeout de leitura em segundos")
    args = parser.parse_args()
    if min(args.tentativas, args.timeout) < 1:
        parser.error("Tentativas e timeout devem ser positivos")
    try:
        preparar(args.destino, args.tentativas, args.timeout)
    except (OSError, ValueError, zipfile.BadZipFile) as erro:
        parser.exit(2, f"Erro: {erro}\n")


if __name__ == "__main__":
    main()
