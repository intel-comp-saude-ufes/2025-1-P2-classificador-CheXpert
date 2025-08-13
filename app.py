
from lib.data import get_data, prepare_img_tensor
from lib.models import MyResnet18
from lib.training import train
from torch import nn
import torch

def print_data_dict(d : dict):
    line_str = '-'*100
    for k, v in d.items():
        print(k)
        print(v)
        print(line_str)

if __name__ == '__main__':
    
    data_dict = get_data()
    train_dataset, test_dataset = data_dict['train_dataset'], data_dict['test_dataset']    

    print(len(data_dict['classes']))
    
    n_classes = len(data_dict['classes'])
    model= MyResnet18(n_classes=n_classes)
    
    model.unfreeze()
    loss_fn= nn.BCEWithLogitsLoss(reduction='mean')
    optimizer= torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    train(model, train_dataset, test_dataset,
          loss_fn, optimizer, save_path='./', device=torch.device('cpu'), verbose=True)