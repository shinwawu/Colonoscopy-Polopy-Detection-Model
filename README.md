# Detecção e segmentação de pólipos em colonoscopia

Treinamento de **YOLO11-seg** e **U-Net** no mesmo conjunto Kvasir-SEG + negativos
HyperKvasir/cecum. O notebook de comparação apresenta métricas, resultados por imagem,
configurações e sobreposições dos dois modelos, no teste interno ou em imagens externas.

## 1. Instalar na máquina que fará o treinamento

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/), abra um terminal
na raiz deste repositório e atualize o código:

```bash
git pull --ff-only
```

Escolha **uma** instalação. Os comandos usam Python 3.12 e as versões do `uv.lock`.
O extra `notebooks` instala JupyterLab; no VS Code também é possível selecionar `.venv`
como kernel usando a extensão Jupyter.

```bash
# Sem GPU NVIDIA, ou para inspecionar os resultados em CPU:
uv sync --frozen --python 3.12 --extra cpu --extra notebooks

# OU: NVIDIA recente, com driver compatível com CUDA 13:
uv sync --frozen --python 3.12 --extra cuda130 --extra notebooks

# OU: NVIDIA com driver compatível com CUDA 12.6:
uv sync --frozen --python 3.12 --extra cuda126 --extra notebooks
```

Linux e Windows x86_64 têm opções explícitas de CPU/CUDA; no macOS é usado o wheel nativo
e o fluxo aqui documentado executa em CPU. CUDA depende de GPU e driver compatíveis.
A configuração segue o [guia de PyTorch do uv](https://docs.astral.sh/uv/guides/integration/pytorch/).

**Nos comandos seguintes, `--no-sync` preserva a variante escolhida acima.** Ao atualizar
as dependências, repita `uv sync` com os mesmos extras; não misture `cpu`, `cuda126` e `cuda130`.
Ao trocar a variante de PyTorch de um ambiente existente, prefira recriar o ambiente para evitar
arquivos compartilhados residuais de instalações CUDA anteriores.
Confira o ambiente antes de treinar:

```bash
uv run --no-sync python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Obter e preparar os dados uma vez

O baseline inclui os dados originais Kvasir-SEG. Novos downloads e dados preparados
ficam locais. O comando abaixo valida ou obtém os 1000 pares imagem/máscara do
[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) em `data/raw/Kvasir-SEG/` e os 1009
negativos de ceco normal do [HyperKvasir](https://github.com/simula/hyper-kvasir)
em `data/raw/hyperkvasir/cecum/`:

```bash
uv run --no-sync python src/deteccao/download_dataset.py
uv run --no-sync python src/deteccao/data_process.py
```

Se preferir baixar o Kvasir-SEG manualmente pela página oficial, extraia-o para ter
`data/raw/Kvasir-SEG/images/*.jpg` e `data/raw/Kvasir-SEG/masks/*.jpg`, ou salve o ZIP
como `data/raw/kvasir-seg.zip` antes de executar o comando. Pastas existentes são validadas
sem baixar novamente. `src/deteccao/download_dataset.py --apenas-negativos` baixa só os negativos.

O download usa timeout longo e até cinco tentativas. Reutiliza a pasta validada ou o ZIP
já baixado e preserva o arquivo para não precisar transferi-lo de novo. Em caso de queda
no meio da transferência, a próxima tentativa reinicia o ZIP; não há retomada por byte.

O preparo cria `data/processed/Kvasir-SEG/` com imagens, máscaras binárias, labels YOLO,
`data.yaml` e `split_manifest.json` (fontes, parâmetros, hashes e associação de cada imagem
a um split). YOLO e U-Net usam **esses mesmos arquivos**. Em cada fonte, frames semelhantes são agrupados
antes da divisão 80/10/10, com seed 42 e correlação >0,90; as proporções podem variar pelo
tamanho dos grupos. Não mude esse conjunto entre os dois treinos.

Se os dados já estão preparados, pule o segundo comando. Para recriá-los conscientemente,
use `src/deteccao/data_process.py --recriar`; isso substitui os dados derivados, preservando as fontes.
A falta de negativos é erro por padrão. `--sem-negativos` cria outro experimento, sem permitir
medir falsos positivos em fundos. Não transfira `data.yaml` de outra máquina: ele é gerado
com o caminho local. Refaça o preparo no destino.

## 3. Treinar os dois modelos, um de cada vez

Com GPU NVIDIA compatível:

```bash
uv run --no-sync python -u src/deteccao/train_yolo.py --device 0 --batch 4
uv run --no-sync python -u src/deteccao/train_unet.py --device 0 --batch 4
```

Para CPU, substitua `--device 0` por `--device cpu`. Sem `--device`, o script escolhe CUDA
quando disponível e CPU caso contrário. Cada treino tem até 100 épocas e early stopping
com paciência de 25 épocas, escolhendo o melhor checkpoint **na validação**; depois avalia
esse checkpoint no teste interno. A U-Net mantém a arquitetura do baseline (base 32),
resolução 384 e loss BCE + Dice; YOLO usa YOLO11s-seg, resolução 640 e inicialização pré-treinada.
Na U-Net, o treino descarta o último batch incompleto quando há pelo menos um batch
completo; validação e teste usam todas as imagens. A seed controla também a augmentation.
No primeiro treino YOLO, o Ultralytics baixa `yolo11s-seg.pt`; se já tiver o arquivo,
use `--modelo weights/yolo11s-seg.pt` para reutilizá-lo.

Em máquinas com pouca memória, comece com `--batch 1 --workers 0`; `--imgsz 320` reduz o custo
do YOLO e `--imgsz 256` o da U-Net. Alterar resolução muda o experimento. O YOLO não usa cache
de imagens em RAM por padrão. `--no-amp` desliga precisão mista. Ajuste o batch à memória
disponível na GPU.

Os nomes padrão separam os dois experimentos:

| Modelo | Checkpoint selecionado | Histórico e configuração | Métricas do teste interno |
| --- | --- | --- | --- |
| YOLO | `runs/yolo-deteccao/weights/best.pt` | `runs/yolo-deteccao/results.csv`, `args.yaml` | `artifacts/yolo-deteccao.json` |
| U-Net | `runs/unet-deteccao/weights/best.pt` | `runs/unet-deteccao/historico.json`, `config.json` | `artifacts/unet-deteccao.json` |

Ambos guardam somente `weights/best.pt`, o melhor checkpoint da validação; o `last.pt` é
descartado. A U-Net inclui configuração nos pesos e grava o histórico a cada época. Uma
interrupção antes da primeira época completa pode não produzir checkpoint. Para repetir um
experimento, escolha `--nome yolo-v2` / `--nome unet-v2` e atualize os caminhos no notebook.
Um nome existente é recusado.

`--runs CAMINHO` e `--artifacts CAMINHO` permitem escolher os diretórios dos resultados.
Os caminhos padrão são os apresentados na tabela.
O diretório de dados é configurado separadamente por `--dados`.

O progresso aparece no terminal. No Linux, para guardar uma cópia, acrescente, por exemplo,
`2>&1 | tee yolo-train.log` ao comando de treino. Use `--help` em cada script para ver os parâmetros.

## 4. Abrir o relatório no notebook

```bash
uv run --no-sync jupyter lab
```

Abra [03_external_image_comparison.ipynb](notebooks/03_external_image_comparison.ipynb)
e execute as células em ordem. Os caminhos padrão já apontam para os checkpoints da tabela.

- Para conferir os resultados após os dois treinamentos: altere `CONJUNTO = "interno"`
  e mantenha `SPLIT_INTERNO = "test"`. As referências são carregadas automaticamente.
- Para imagens de fora: mantenha `CONJUNTO = "externo"`, configure `ORIGEM` e, se houver,
  `ANOTACOES`, conforme a seção abaixo.
- `DEVICE = "cpu"` permite visualizar em uma máquina sem CUDA; `"cuda:0"` usa a GPU.

**O próprio notebook é o relatório:** mostra resumo das métricas, tabela por imagem,
configuração, procedência e painéis original/referência/YOLO/U-Net. Também exporta CSV, JSON,
máscaras e sobreposições para uma nova pasta em `runs/comparacao_externa_<data>/`
(ou `comparacao_interna_<split>_<data>/`, no modo interno).
Sem imagens ou pesos ele informa as pendências. Os arquivos de pesos não vêm no Git:
precisam ser gerados pelos treinamentos ou fornecidos por quem os treinou.

Os outros notebooks são [01_dataset_inspection.ipynb](notebooks/01_dataset_inspection.ipynb)
e [02_image_similarity.ipynb](notebooks/02_image_similarity.ipynb). O segundo explora a
similaridade dos frames; o limiar usado no preparo é explicitamente 0,90.

## 5. Adicionar imagens externas

Coloque imagens em `data/external/images/` (subpastas são aceitas), separadas dos dados internos.
Para análise visual, não é necessário CSV. Para medir resultados, copie
[examples/anotacoes.exemplo.csv](examples/anotacoes.exemplo.csv) para
`data/external/anotacoes.csv`, substitua as linhas pelos seus arquivos e configure no notebook:

```python
CONJUNTO = "externo"
ORIGEM = ROOT / "data/external/images"
ANOTACOES = ROOT / "data/external/anotacoes.csv"
```

O CSV tem as colunas `imagem,mascara,presenca,origem`. `imagem` é relativa à pasta de entrada;
`mascara` é relativa à pasta do CSV. Máscaras devem ter as dimensões da imagem original,
fundo 0 e pólipos >0.

| Anotação | Avaliação disponível |
| --- | --- |
| Máscara de referência | Dice/IoU e presença por imagem |
| Sem máscara, `presenca=0` | Negativo confirmado; conta falsos positivos |
| Sem máscara, `presenca=1` | Presença por imagem; sem Dice/IoU |
| Sem máscara e presença vazia | Somente inspeção visual |

Imagem sem anotação não é tratada como negativa. Máscara citada e ausente, dimensões incorretas
ou presença contraditória são erros. Dice/IoU são médias sobre positivos com máscara, incluindo
falhas de detecção com zero; fundos não entram nessa média. Sensibilidade e falsos positivos
medem presença de alguma região por imagem, sem garantir localização correta da lesão.
O resumo também preserva mediana, erro padrão amostral e contagens de Dice <0,5 e >0,9
do comparador original. O erro padrão fica vazio com menos de duas máscaras positivas.

Os dois modelos usam a mesma lista de imagens, grade de avaliação 384×384 e filtro de área
mínima de 500 pixels nessa grade. A U-Net lê do checkpoint sua resolução e normalização.
A máscara da U-Net é projetada diretamente da saída da rede para a grade de avaliação;
o redimensionamento à imagem original é usado somente nas exportações visuais.
`AREA_MINIMA = 0` permite um experimento sem filtro. Fixe limiares usando validação separada,
registre a origem das imagens e verifique sobreposição/similaridade com os dados internos antes
de chamar um resultado de externo. Máscaras ausentes não permitem comparar segmentação quantitativamente.

## Alternativa pelo terminal

O notebook é a interface principal. Para exportar a comparação sem abrir Jupyter:

```bash
# Teste interno, usando os dois checkpoints padrão:
uv run --no-sync python src/deteccao/compare_models.py

# Conjunto externo anotado:
uv run --no-sync python src/deteccao/compare_models.py --origem data/external/images --anotacoes data/external/anotacoes.csv

# Um modelo por vez, se necessário:
uv run --no-sync python src/deteccao/predict_yolo.py --origem data/external/images
uv run --no-sync python src/deteccao/predict_unet.py --origem data/external/images
```

Os dados e resultados versionados em `data/`, `runs/` e `artifacts/` registram o baseline
do projeto. O `.gitignore` exclui novos dados derivados, execuções, pesos, ambientes
e caches. Cada experimento deve usar um nome próprio para preservar os resultados
anteriores. Os checkpoints treinados não estão incluídos nos relatórios de métricas.
O carregador da U-Net aceita tanto o `state_dict` do baseline (base 32 e resolução 384)
quanto checkpoints que incluem a configuração da rede.

O notebook de comparação também apresenta as métricas do baseline registradas em
`artifacts/comparacao.json`, identificadas como resultados do experimento original.
Novas avaliações exigem os checkpoints e as imagens configurados no notebook.

Para compartilhar um relatório executado, salve uma cópia do notebook e confira as imagens
que suas saídas incorporam antes de versioná-lo.
