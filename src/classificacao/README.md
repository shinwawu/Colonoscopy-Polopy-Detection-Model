# Cenário: classificação do tipo de pólipo

Este cenário ainda não existe como código. Enquanto o `src/deteccao/` responde
**"há pólipo, e onde está"**, este responderia **"que tipo de pólipo é"** — várias
classes em vez de uma.

Ele está vazio de propósito: manter 13 arquivos duplicados que não executam só
dificulta encontrar o que importa. O código volta quando houver dado.

## Por que não dá para começar hoje

Nenhuma fonte do repositório tem rótulo de tipo:

| fonte | o que tem |
| --- | --- |
| `data/raw/Kvasir-SEG` | 1000 imagens, máscara binária, **uma classe** (`polyp` nas 1071 caixas do `kavsir_bboxes.json`) |
| `data/raw/hyperkvasir/cecum` | 1009 frames de mucosa normal, usados como negativos |

O tipo do pólipo é **diagnóstico histológico**, confirmado por biópsia — não é
derivável das imagens que você tem.

## Fontes possíveis

| dataset | rótulo | acesso |
| --- | --- | --- |
| KUMC / PolypsSet | adenomatoso, hiperplásico | público (figshare) |
| PICCOLO | histologia + classificação de Paris | solicitação formal |
| SUN Database | patologia | solicitação formal |

## O que fazer quando houver dado

1. Copiar os scripts de `src/deteccao/` para cá
2. Em `data_process.py`, `train_unet.py` e `train_yolo.py`: ajustar `CLASSES`, `DADOS` e `NOME`
   (não há `config.py`; cada script carrega a própria configuração)
3. Adaptar o preprocess para ler o tipo de cada imagem e gerar o `.txt` do YOLO
   com o índice da classe correta (hoje todos saem como classe 0)
4. A U-Net é binária: `n_classes=len(CLASSES)` e CrossEntropy no lugar de BCE + Dice

## Expectativa

O desempenho publicado nessa tarefa é bem menor que em detecção. Mesmo os datasets
com esse rótulo carregam incerteza, porque a aparência endoscópica nem sempre
distingue adenoma de hiperplásico. Não é a mesma dificuldade do cenário de detecção.
