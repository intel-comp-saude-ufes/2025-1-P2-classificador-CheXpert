
from lib.data import get_data, get_data_chest_x_ray_image
from lib.training import train
from lib.metrics import MetricCollection
from lib.utils import get_device
from lib.models import MyResnet18
import torch

def print_data_dict(d : dict):
    line_str = '-'*100
    for k, v in d.items():
        print(k)
        print(v)
        print(line_str)

if __name__ == '__main__':
    

    chest_x_ray_dict = get_data_chest_x_ray_image()
    
    train_dataset, val_dataset, test_dataset = chest_x_ray_dict['train_dataset'], chest_x_ray_dict['val_dataset'], chest_x_ray_dict['test_dataset']
    
    device = get_device()
    n_classes = len(chest_x_ray_dict['classes'])
    metric_collection = MetricCollection(device=device, num_classes=n_classes, task_type='multiclass')
    
    model = MyResnet18(n_classes=n_classes)    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam()
    
    model.unfreeze()
    
    train(model, train_dataset, val_dataset)