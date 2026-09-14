# Detecção de pólipos em colonoscopia

Segmentação de pólipos com YOLO11, treinada no [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
com imagens de mucosa normal do [HyperKvasir](https://github.com/simula/hyper-kvasir) como negativos.

## Requisitos

```bash
uv sync
```

GPU com CUDA é recomendada (o treino usa `device=0`).

## Dados

As imagens e máscaras do Kvasir-SEG já vêm versionadas em `data/raw/Kvasir-SEG/`.


```bash
mkdir -p data/raw/hyperkvasir
curl -L --progress-bar \
  "https://files.osf.io/v1/resources/mh9sj/providers/osfstorage/5dfb5b530236b8000c746a8e/?zip=" \
  -o data/raw/hyperkvasir/cecum.zip

unzip -q data/raw/hyperkvasir/cecum.zip -d data/raw/hyperkvasir/cecum
ls data/raw/hyperkvasir/cecum | wc -l    # deve dar 1009
rm data/raw/hyperkvasir/cecum.zip
```

