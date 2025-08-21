from torch import nn
import torch
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision.models import mobilenet_v3_large, mobilenet_v2, MobileNet_V3_Large_Weights, MobileNet_V2_Weights
from torchvision.models import densenet121, densenet161, DenseNet121_Weights, DenseNet161_Weights
from torchvision.models import vit_b_16, vit_b_32, ViT_B_16_Weights, ViT_B_32_Weights

class FreezableCNN(nn.Module):
    '''
    Classe base utilizada para definir as funcionalidades de freeze e unfreeze das redes neurais pré-treinadas encontradas no torchvision
    '''
    
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = None
        self.classifier = None

    def forward(self, X):
        features = self.feature_extractor(X)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits

    def freeze(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True


class MyResnet(FreezableCNN):
    def __init__(self, resnet_version: str, n_classes=2):
        super().__init__()
        self.version = resnet_version

        if resnet_version == 'resnet18':
            modules = list(resnet18(weights=ResNet18_Weights.DEFAULT).children())
        elif resnet_version == 'resnet34':
            modules = list(resnet34(weights=ResNet34_Weights.DEFAULT).children())
        else:
            raise ValueError(f'invalid resnet version `{resnet_version}`')

        self.feature_extractor = nn.Sequential(*modules[:-1])
        self.classifier = nn.LazyLinear(n_classes)


class MyMobileNet(FreezableCNN):
    def __init__(self, mobilenet_version: str, n_classes=2):
        super().__init__()
        self.version = mobilenet_version

        if mobilenet_version == 'mobilenet_v3_large':
            modules = list(mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).children())
        elif mobilenet_version == 'mobilenet_v2':
            modules = list(mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).children())
        else:
            raise ValueError(f'invalid mobilenet version `{mobilenet_version}`')

        self.feature_extractor = nn.Sequential(*modules[:-1])
        self.classifier = nn.LazyLinear(n_classes)


class MyDenseNet(FreezableCNN):
    def __init__(self, densenet_version: str, n_classes=2):
        super().__init__()
        self.version = densenet_version

        if densenet_version == 'densenet121':
            modules = list(densenet121(weights=DenseNet121_Weights.DEFAULT).children())
        elif densenet_version == 'densenet161':
            modules = list(densenet161(weights=DenseNet161_Weights.DEFAULT).children())
        else:
            raise ValueError(f'invalid densenet version `{densenet_version}`')

        self.feature_extractor = nn.Sequential(*modules[:-1])
        self.classifier = nn.LazyLinear(n_classes)
        
        
class MyViT(nn.Module):
    def __init__(self, vit_version: str, n_classes=2):
        super().__init__()
        self.version = vit_version

        if vit_version == 'vit_b_16':
            self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif vit_version == 'vit_b_32':
            self.vit = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        else:
            raise ValueError(f'Invalid ViT version `{vit_version}`')

        self.vit.heads = nn.Identity()  
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x):
        features = self.vit(x)  
        logits = self.classifier(features)
        return logits

    def freeze(self, upto_layer=None):
        """
        Congela até determinada camada do ViT.
        Se upto_layer=None, congela tudo.
        """
        for idx, param in enumerate(self.vit.parameters()):
            if upto_layer is None or idx <= upto_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze(self):
        for param in self.vit.parameters():
            param.requires_grad = True


def freeze_feature_extractor(model: nn.Module, head_attr: str = 'fc', frozen: bool = True):
    '''
    recebe um modelo e o atributo referente a camada de classificação, se frozen=True, congela a parte convolucional e descongela a camada de classificação. 
    Do contrário, congela tudo.
    '''
    
    for param in model.parameters(): ## se é pra congelar, desativa
        param.requires_grad = not frozen

    ## o nome do classificador pode mudar dependendo do modelo
    if hasattr(model, head_attr):
        head = getattr(model, head_attr)
        for param in head.parameters(): ## no entanto mantém a última camada (classifier) treinável
            param.requires_grad = True
    else:
        raise ValueError(f"Model does not have the attribute '{head_attr}'")
    
    return model


## last layer from resnet18 Linear(in_features=512, out_features=1000, bias=True)
def get_resnet18(n_classes : int, frozen=False):
    '''
    Retorna uma rede Resnet18 pré-treinada com camada de classficação substituida para uma adequada ao número de classes do problema
    '''
    
    resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
     
    print(list(resnet18_model.children()))
        
    new_fc = nn.LazyLinear(n_classes)
    resnet18_model.fc = new_fc
    
    #print(list(resnet18_model.children()))
    
    resnet18_model = freeze_feature_extractor(resnet18_model, 'fc', frozen=frozen)
    
    return resnet18_model