import kagglehub
import pandas as pd
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchvision.io import read_image

from torch.utils.data import Subset, random_split
from sklearn.model_selection import StratifiedKFold


########################################################################

def get_path_chexpert():
    return kagglehub.dataset_download("ashery/chexpert")

def get_path_chestxray():
    return kagglehub.dataset_download("alsaniipe/chest-x-ray-image")

########################################################################


class CheXpertDataSet(Dataset):
    def __init__(self, data_df, label_columns, path_column, transformation=None):
        self.label_columns = label_columns
        self.path_column = path_column
        
        self.path_series = data_df[path_column]
        self.label_data = torch.tensor(data_df[label_columns].values, dtype=torch.float32)
        
        self.transformation = transformation
               
        return 
    
    def __len__(self):
        return len(self.path_series)
    
    def __getitem__(self, index):
        img = read_image(self.path_series.iloc[index]) ## é mais eficiente
        
        if self.transformation:
            img = self.transformation(img)
            
        y = self.label_data[index]        
        
        return img, y
    
    def to_dataloader(self, batch_size=64, shuffle=False):
        dataset = self
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def to_rgb_if_needed(x):
    if x.shape[0] == 1: ## Converte para RGB caso for gray scale
        return x.repeat(3, 1, 1)
    return x

def get_transformations(img_size = (224, 224)):
    
    train_transform = v2.Compose([
        v2.ToImage(),  # 1. Converte a imagem PIL para um tensor
        v2.Lambda(to_rgb_if_needed), # 2. Garante que o tensor tenha 3 canais
        v2.Resize(img_size), # 3. Redimensiona a imagem
        v2.ToDtype(torch.float32, scale=True), # 4. Converte para float e normaliza para o intervalo [0, 1]
        ####### Data Augmentation #######
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        v2.RandomPerspective(0.1),
        ################################
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 5. Normaliza os dados
    ])
    
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Lambda(to_rgb_if_needed),
        v2.Resize(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return (train_transform, test_transform)

def fix_df_path(df:pd.DataFrame, root_path: str):
    prefix = 'CheXpert-v1.0-small/'
    fix_path = lambda _path: os.path.join(root_path, str(_path).replace(prefix, ''))
    result_series = df['Path'].apply(fix_path)
    return result_series
    
def treat_df_label_columns(df: pd.DataFrame, label_columns: list[str], inplace=False):
    df = df if inplace else df.copy()
    df[label_columns] = df[label_columns].fillna(0)
    df[label_columns] = df[label_columns].replace({-1:0})
    return df 

def prepare_img_tensor(img_t, mean, std):
    '''
    Desfaz transformações aplicadas na imagem para plot com matplotlib
    '''
    
    mean = torch.tensor(mean, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    img_t = img_t * std + mean
    img_t = img_t.clamp(0, 1)
    img_t = img_t.permute(1, 2, 0)  # (H, W, C)
    return img_t.cpu().numpy()

def encode_categoric_column(df: pd.DataFrame, c_name: str, data_dict):
    unique_labels = df[c_name].unique()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx:label for idx, label in enumerate(unique_labels)}
    data_dict[c_name] = {'label2id' : label2id, 'idtolabel' : id2label}
    return

def filter_frontal_images(df:pd.DataFrame, position_column):
    mask = df[position_column] == 'Frontal'
    df = df[mask]
    return df

def filter_data_frame(df:pd.DataFrame, size=0.1):
    final_size = int(df.shape[0]*size)
    df = df.iloc[:final_size]
    return df

'''
    -1 --> inconclusivo
    none --> dado faltante
    
    Especificação da coluna AP/PA
        AP --> raio x entra pela frente e sai atrás
        PA --> raio x entra por trás e sai na frente
        
        No Finding -> A coluna No Finding é um indicador de ausência de todas as outras patologias. Ela é derivada dos demais rótulos 
            não tem valor clínico como informação extra a ser passada junto com a imagem.
            Pode ser usada pra transformar o problema em classificação binária com No Finding = 1(normal), No Finding = 0 (anormal)
'''


def get_chexpert() -> dict:
    data_dict = dict()
    
    PATH_CHEXPERT = get_path_chexpert()
    
    train_transformation, test_transformation = get_transformations()
    
    train_csv_path = os.path.join(PATH_CHEXPERT, 'train.csv')
    valid_csv_path = os.path.join(PATH_CHEXPERT, 'valid.csv')
    df_train = pd.read_csv(train_csv_path)    
    df_valid = pd.read_csv(valid_csv_path)
    
    ## remove 'Frontal/Lateral' column after filter just the Frontal Images
    pos_col = 'Frontal/Lateral'
    frontal_str = 'Frontal'
    df_train = df_train[df_train[pos_col] == frontal_str]
    df_valid = df_valid[df_valid[pos_col] == frontal_str]
    
    df_train = df_train.drop(labels=[pos_col], axis=1)
    df_valid = df_valid.drop(labels=[pos_col], axis=1)
    
    metadata_columns = ['Sex', 'Age', 'AP/PA']
    path_column = 'Path'
    label_columns = [c for c in df_train.columns if c not in (metadata_columns + [path_column])]
    data_dict['classes'] = label_columns
    data_dict['idx_to_class'] = {idx:label for idx,label in enumerate(label_columns)}
    data_dict['class_to_idx'] = {label: idx for idx,label in enumerate(label_columns)}
    
    ## metadata_columns : Sex, Age, AP/PA
    categoric_columns = ['Sex', 'AP/PA'] 
    for c in categoric_columns:
        encode_categoric_column(df_train, c, data_dict)
        df_train[c] = df_train[c].replace(data_dict[c]['label2id'])

    ## fixing paths 
    df_train['Path'] = fix_df_path(df_train, PATH_CHEXPERT)
    df_valid['Path'] = fix_df_path(df_valid, PATH_CHEXPERT)
    
    ## setting nan -> 0, and -1 -> 0, for all label_columns
    df_train = treat_df_label_columns(df_train, label_columns, inplace=True)
    df_valid = treat_df_label_columns(df_valid, label_columns, inplace=True)
    
    df_train = filter_data_frame(df_train, size=1.00) ## filtrando pra testar
    #print('lines:',df_train.shape[0])
    
    df_train = df_train.reset_index(drop=True) # this is needed, do not remove this reset index lines (DataSet stores pandas Series, so they must have the correct indexes)
    df_valid = df_valid.reset_index(drop=True)
    
    ## there is no Path or Age with nan value both in train and valid dataframes !!!
    
    train_dataset = CheXpertDataSet(df_train, label_columns, path_column, transformation=train_transformation)
    test_dataset = CheXpertDataSet(df_valid, label_columns, path_column, transformation=test_transformation)
    
    data_dict['train_dataset'] = train_dataset
    data_dict['test_dataset'] = test_dataset
    data_dict['classes'] = label_columns
    
    data_dict['df_train'] = df_train
    data_dict['df_valid'] = df_valid
    
    return data_dict


#---------------------------------------------------------

class TransformDataset(Dataset):
    def __init__(self, subset, transform=None, class_map=None):
        self.subset = subset
        self.transform = transform
        self.class_map=class_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        
        if self.class_map:
            label = self.class_map[label]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


def my_random_split(dataset: ImageFolder, size=float) -> tuple[Subset, Subset]:
    '''
    random test split que utiliza o do pytorch como auxiliar, devolve subsets (referências) de dataset maior, não há redundancia de dados.
    '''
    
    train_size = int((size) * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    
    return (train_subset, val_subset)


## https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image
def get_data_chest_x_ray_image(img_size=(224, 224), split_ratio=0.8, kfold=None, seed=42):
    """
    Carrega dataset de Raio-X do Kaggle e retorna dados no formato:
      - Modo normal: retorna train_dataset e val_dataset já com transformações
      - Modo K-Fold: retorna base_dataset, lista de folds e metadados
    """
    # Baixar dados
    PATH_CHESTXRAY = get_path_chestxray()
    
    train_path = os.path.join(PATH_CHESTXRAY, 'Data/train')
    test_path = os.path.join(PATH_CHESTXRAY, 'Data/test')

    base_train_dataset = ImageFolder(root=train_path)
    train_transform, test_transform = get_transformations(img_size=img_size)
    test_dataset = ImageFolder(root=test_path, transform=test_transform)

    idx_to_class = {v: k for k, v in base_train_dataset.class_to_idx.items()}

    ### modo normal, divide com random split
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

    ## modo kfold, usa StratifiedKfold para a divisão dos índices e devolve em data_dict a lista de índices para cada fold
    else:
        targets = [y for _, y in base_train_dataset]  # labels para stratificação
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        folds = list(skf.split(range(len(base_train_dataset)), targets))

        return {
            'mode': 'kfold',
            'base_dataset': base_train_dataset,
            'folds': folds,  # lista de (train_idx, val_idx)
            'train_transform': train_transform,
            'test_transform': test_transform,
            'test_dataset': test_dataset,
            'idx_to_class': idx_to_class,
            'class_to_idx': base_train_dataset.class_to_idx,
            'classes': base_train_dataset.classes
        }