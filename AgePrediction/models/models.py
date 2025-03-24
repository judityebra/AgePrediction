import torch
import torch.nn as nn
import torch.nn.functional as F



def conv3x3(in_planes, out_planes, stride=1):
    """
    Realitza una operació de convolució amb un kernel de mida 3x3 i padding.

    Parameters:
    - in_planes (int): Dimensió de l'entrada.
    - out_planes (int): Dimensió de la sortida.
    - stride (int, opcional): Pas de la convolució. Per defecte 1.

    Returns:
    - nn.Conv2d: una capa de convolució.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# Definició d'un bloc bàsic de la ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Inicialitza un bloc bàsic de la ResNet.

        Parameters:
        - inplanes (int): Dimensió de l'entrada.
        - planes (int): Dimensió de la sortida.
        - stride (int, opcional): Pas de la convolució. Per defecte 1.
        - downsample (callable, opcional): Funció per a la downsampling de l'entrada.
        """

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Aplica el bloc bàsic a x.

        Parameters:
        - x (Tensor): Entrada del bloc bàsic.

        Returns:
        - out: Sortida del bloc bàsic.
        """
                
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Definició de la classe ResNetCe
class ResNetCe(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        """
        Inicialitza una nova xarxa ResNet amb loss Cross-Entropy.

        Parameters:
        - block (type): Bloc bàsic utilitzat en la construcció de la xarxa.
        - layers (list[int]): Llista que indica el nombre de capes residuals en cada bloc de la xarxa.
        - num_classes (int): Nombre total de classes per a la classificació.
        - grayscale (bool): Indica si les imatges d'entrada són en escala de grisos.
        """
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNetCe, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

        # Inicialització de pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Crea un bloc de capes residuals.

        Parameters:
        - block (type): Bloc bàsic utilitzat en la construcció de la xarxa.
        - planes (int): Dimensió de la sortida de les capes residuals.
        - blocks (int): Nombre de capes residuals en el bloc.
        - stride (int): Pas de la convolució.

        Returns:
        - nn.Sequential: Seqüència de capes residuals.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Aplica la xarxa ResNet amb Cross-Entropy a x i retorna les prediccions.

        Parameters:
        - x (Tensor): Entrada de la xarxa.

        Returns:
        - logits (Tensor): Prediccions de la xarxa.
        - probas (Tensor): Probabilitats de les prediccions.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


# Definició de la classe ResNetOrdinal
class ResNetOrdinal(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        """
        Inicialitza una nova xarxa ResNet amb loss Ordinal.

        Parameters:
        - block (type): Bloc bàsic utilitzat en la construcció de la xarxa.
        - layers (list[int]): Llista que indica el nombre de capes residuals en cada bloc de la xarxa.
        - num_classes (int): Nombre total de classes per a la classificació.
        - grayscale (bool): Indica si les imatges d'entrada són en escala de grisos.
        """
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNetOrdinal, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, (self.num_classes - 1) * 2)

        # Inicialització de pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Crea un bloc de capes residuals.

        Parameters:
        - block (type): Bloc bàsic utilitzat en la construcció de la xarxa.
        - planes (int): Dimensió de la sortida de les capes residuals.
        - blocks (int): Nombre de capes residuals en el bloc.
        - stride (int): Pas de la convolució.

        Returns:
        - nn.Sequential: Seqüència de capes residuals.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Aplica la xarxa ResNet amb Ordinal a x i retorna les prediccions.

        Parameters:
        - x (Tensor): Entrada de la xarxa.

        Returns:
        - logits (Tensor): Prediccions de la xarxa.
        - probas (Tensor): Probabilitats de les prediccions.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits.view(-1, (self.num_classes - 1), 2)
        probas = F.softmax(logits, dim=2)[:, :, 1]
        return logits, probas


# Definició de la classe ResNetCoral
class ResNetCoral(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        """
        Inicialitza una nova xarxa ResNet amb loss Coral.

        Parameters:
        - block (type): Bloc bàsic utilitzat en la construcció de la xarxa.
        - layers (list[int]): Llista que indica el nombre de capes residuals en cada bloc de la xarxa.
        - num_classes (int): Nombre total de classes per a la classificació.
        - grayscale (bool): Indica si les imatges d'entrada són en escala de grisos.
        """
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNetCoral, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes - 1).float())

        # Inicialització de pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Crea un bloc de capes residuals.

        Parameters:
        - block (type): Bloc bàsic utilitzat en la construcció de la xarxa.
        - planes (int): Dimensió de la sortida de les capes residuals.
        - blocks (int): Nombre de capes residuals en el bloc.
        - stride (int): Pas de la convolució.

        Returns:
        - nn.Sequential: Seqüència de capes residuals.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Aplica la xarxa ResNet amb Coral a x i retorna les prediccions.

        Parameters:
        - x (Tensor): Entrada de la xarxa.

        Returns:
        - logits (Tensor): Prediccions de la xarxa.
        - probas (Tensor): Probabilitats de les prediccions.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas




def resnet34(num_classes, grayscale, loss='ce'):
    """
    Construeix un model ResNet-34 per a diferents tipus de loss.

    Parameters:
    - num_classes (int): Nombre de classes per a la classificació.
    - grayscale (bool): Indica si les imatges d'entrada són en escala de grisos.
    - loss (str, opcional): Tipus de pèrdua a utilitzar. Per defecte Cross-Entropy.

    Returns:
    - nn.Module: Instància de la xarxa ResNet-34 configurada segons el tipus de loss.
    """

    if loss == 'ce':
        model = ResNetCe(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=grayscale)
    elif loss == 'coral':
        model = ResNetCoral(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=grayscale)
    elif loss == 'ordinal':
        model = ResNetOrdinal(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=grayscale)
    else:
        raise ValueError("Pèrdua incorrecta introduïda.")
    return model

# Aquest codi defineix tres variacions d'una xarxa ResNet per diferents tipus de pèrdua: 
# ResNetCe per la pèrdua de classificació creuada,
# ResNetOrdinal per la classificació ordinal,
# i ResNetCoral per la regressió ordinal. Les arquitectures són similars en estructura, amb diferències en les capes finals per adaptar-se als diferents tipus de tasques.

