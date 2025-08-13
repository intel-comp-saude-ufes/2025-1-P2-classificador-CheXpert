from torch import nn
import torch
from torchvision.models import resnet18, ResNet18_Weights

class MyResnet18(nn.Module):
    def __init__(self, n_classes=2):
        super(MyResnet18, self).__init__()
        
        modules = list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())
        self.feature_extractor = nn.Sequential(*modules[:-1])
        self.classifier = nn.LazyLinear(n_classes)

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