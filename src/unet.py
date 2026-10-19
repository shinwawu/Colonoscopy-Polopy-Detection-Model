
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (conv 3x3 -> BatchNorm -> ReLU) x2 
    conv para reduzir o tamanho da imagem e aumentar o numero de canais,
    batchnotm para normalizar a saida da convolucao, e ReLU função atividora para nao linearidade.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # ele serve como um wrapper para o bloco sequencial, para que possamos chamá-lo como uma camada
        return self.block(x)


class Down(nn.Module):
    """
    serve como um encoder: reduz a resolucao e aumenta o numero de canais, para que o decoder possa reconstruir a imagem.
    cada descida consiste em um maxpool 2x2 (reduz a resolucao pela metade) seguido de duas convolucoes 3x3 (aumenta o numero de canais).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    serve como um decoder: aumenta a resolucao e reduz o numero de canais, para que a imagem reconstruida tenha a mesma resolucao da original.
    cada subida consiste em uma convolucao transposta 2x2 (aumenta a resolucao pela metade) seguida de duas convolucoes 3x3 (reduz o numero de canais).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d aprende como ampliar, o numero de canais cai pela metade
        # para que, apos concatenar a skip (do mesmo tamanho), voltemos ao mesmo tamanho de canais da entrada
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # se o lado da entrada nao for divisivel por 16, o arredondamento do pooling
        # faz o upsample errar por 1 pixel; o pad realinha antes de concatenar
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    U-Net de profundidade 4
    4 downsamplings, 4 upsamplings, e uma camada de convolucao 1x1 no final para reduzir o numero de canais para o numero de classes.
    """

    def __init__(self, n_channels=3, n_classes=1, base=64):
        super().__init__()
        # a largura da U-Net e o numero de canais da primeira camada convolucional, que dobra a cada downsampling
        c = [base, base * 2, base * 4, base * 8, base * 16]

        self.inc = DoubleConv(n_channels, c[0])
        self.down1 = Down(c[0], c[1])
        self.down2 = Down(c[1], c[2])
        self.down3 = Down(c[2], c[3])
        self.down4 = Down(c[3], c[4])        

        self.up1 = Up(c[4], c[3])
        self.up2 = Up(c[3], c[2])
        self.up3 = Up(c[2], c[1])
        self.up4 = Up(c[1], c[0])

        
        self.outc = nn.Conv2d(c[0], n_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        """Sem pesos pre-treinados, a inicializacao passa a importar de verdade."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # inicializacao de He (ou Kaiming) para ReLU, que e a funcao de ativacao usada na U-Net
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # o forward passa a imagem pela U-Net, retornando os logits da segmentacao
        # cada nivel do encoder reduz a resolucao e aumenta o numero de canais, enquanto cada nivel do decoder aumenta a resolucao e reduz o numero de canais, usando as skip connections para manter o detalhe espacial
        # a saida final e uma convolucao 1x1 que reduz o numero de canais para o numero de classes, sem alterar a resolucao da imagem
        # LOGITS, nao probabilidade: a BCEWithLogitsLoss aplica a sigmoide internamente, de forma numericamente estavel.
        x0 = self.inc(x)   
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        y = self.up1(x4, x3)    # cada subida recebe a skip do nivel correspondente
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        y = self.up4(y, x0)

        #  BCEWithLogitsLoss aplica a sigmoide
        # internamente, de forma numericamente estavel.
        return self.outc(y)
