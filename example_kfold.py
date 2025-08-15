from torch.utils.data import random_split, Subset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
import os
import kagglehub
from torch.utils.data import Dataset, DataLoader

class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transformations(img_size = (224, 224)):
    from torchvision.transforms import v2
    import torch
    '''
    contém a definição das transformações utilizadas nesse trabalho
    '''
    
    train_transform = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")), ## colocando em rgb
        v2.Resize(img_size),
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        #######
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        v2.RandomPerspective(0.1),
        ########
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    test_transform = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")), ## colocando em rgb
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.Resize(img_size),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return (train_transform, test_transform)


def my_random_split(dataset: ImageFolder, size: float):
    """
    Faz split aleatório em treino/val.
    """
    train_size = int(size * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

def get_data_chest_x_ray_image(img_size=(224, 224), split_ratio=0.8, kfold=None, seed=42):
    """
    Carrega dataset de Raio-X do Kaggle e retorna dados no formato:
      - Modo normal: retorna train_dataset e val_dataset já com transformações
      - Modo K-Fold: retorna base_dataset, lista de folds e metadados
    """
    # Baixar dados
    PATH = kagglehub.dataset_download("alsaniipe/chest-x-ray-image")    
    train_path = os.path.join(PATH, 'Data/train')
    test_path = os.path.join(PATH, 'Data/test')

    base_train_dataset = ImageFolder(root=train_path)
    train_transform, test_transform = get_transformations(img_size=img_size)
    test_dataset = ImageFolder(root=test_path, transform=test_transform)

    idx_to_class = {v: k for k, v in base_train_dataset.class_to_idx.items()}

    # -------------------------
    # MODO NORMAL (sem K-Fold)
    # -------------------------
    if kfold is None:
        train_subset, val_subset = my_random_split(base_train_dataset, size=split_ratio)
        train_dataset = TransformDataset(train_subset, train_transform)
        val_dataset = TransformDataset(val_subset, test_transform)

        return {
            'mode': 'normal',
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'idx_to_class': idx_to_class,
            'class_to_idx': base_train_dataset.class_to_idx,
            'classes': base_train_dataset.classes
        }

    # -------------------------
    # MODO K-FOLD
    # -------------------------
    else:
        targets = [y for _, y in base_train_dataset]  # labels para stratificação
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        folds = list(skf.split(range(len(base_train_dataset)), targets))

        return {
            'mode': 'kfold',
            'base_dataset': base_train_dataset,
            'folds': folds,  # lista de (train_idx, val_idx)
            'train_transform': train_transform,
            'val_transform': test_transform,
            'test_dataset': test_dataset,
            'idx_to_class': idx_to_class,
            'class_to_idx': base_train_dataset.class_to_idx,
            'classes': base_train_dataset.classes
        }
