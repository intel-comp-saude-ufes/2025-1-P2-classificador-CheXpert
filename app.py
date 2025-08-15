
from lib.data import get_data, get_data_chest_x_ray_image
from lib.training import train
from lib.metrics import MetricCollection
from lib.utils import get_device
from lib.models import MyResnet18
import torch
import torchmetrics
import os

def print_data_dict(d : dict):
    line_str = '-'*100
    for k, v in d.items():
        print(k)
        print(v)
        print(line_str)
        
        
def get_my_metrics(device):
    metrics = MetricCollection(device=device)
    metrics.register('accuracy', torchmetrics.Accuracy(task='multiclass', num_classes=n_classes))
    metrics.register('precision', torchmetrics.Precision(task='multiclass', num_classes=n_classes, average='macro'))
    metrics.register('recall', torchmetrics.Recall(task='multiclass', num_classes=n_classes, average='macro'))
    metrics.register('f1', torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average='macro'))
    
    return metrics


if __name__ == '__main__':
    

    chest_x_ray_dict = get_data_chest_x_ray_image()
    

    train_dataset, val_dataset, test_dataset = chest_x_ray_dict['train_dataset'], chest_x_ray_dict['val_dataset'], chest_x_ray_dict['test_dataset']
    
    device = get_device()
    n_classes = len(chest_x_ray_dict['classes'])
    
    metric_collection = get_my_metrics(device=device)
    model = MyResnet18(n_classes=n_classes).to(device=device)    
    model.unfreeze()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    
    
    num_workers = max(1,os.cpu_count()-1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    
    train(model, train_dataloader, val_dataloader, loss_fn, optimizer, save_path='./', device=device, metrics=metric_collection)