import kagglehub
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


class CheXpertDataSet(Dataset):
    def __init__(self, data_df, label_columns, meta_columns, path_column, transformation=None):
        self.label_columns = label_columns
        self.meta_columns = meta_columns
        self.path_column = path_column
        
        self.path_series = data_df[path_column] ## it is more eficient to save strings on Series or python lists than in nparrays
        self.meta_data = data_df[meta_columns].to_numpy()
        self.label_data = data_df[label_columns].to_numpy()
        
        self.transformation=transformation
        return 
    def __len__(self):
        return self.data_df.shape[0]
    def __getitem__(self, index):
        img = Image.open(self.path_series[index])
        if self.transformation:
            img = self.transformation(img)
            
        y = self.label_data[index]        
        metadata = self.meta_data[index]
        return metadata, y, img


## PATH = '/home/msmartin/.cache/kagglehub/datasets/ashery/chexpert/versions/1'


def get_transformations(img_size = (224, 224)):
    
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
    mean = torch.tensor(mean, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    img_t = img_t * std + mean
    img_t = img_t.clamp(0, 1)
    img_t = img_t.permute(1, 2, 0)  # (H, W, C)
    return img_t.cpu().numpy()

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

def encode_categoric_column(df: pd.DataFrame, c_name: str, data_dict):
    unique_labels = df[c_name].unique()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx:label for idx, label in enumerate(unique_labels)}
    data_dict[c_name] = {'label2id' : label2id, 'idtolabel' : id2label}
    return


## TODO: remover as instâncias do dataset que não são Frontal/Lateral == Frontal (Tem 90% de imagens frontais e 10% de laterais, são problemas diferentes)
## e está desbalanceado
def read_data():
    import kagglehub
    path = kagglehub.dataset_download("ashery/chexpert")
    
    data_dict = dict()
    
    train_transformation, test_transformation = get_transformations()
    
    train_csv_path = os.path.join(path, 'train.csv')
    valid_csv_path = os.path.join(path, 'valid.csv')
    df_train = pd.read_csv(train_csv_path)    
    df_valid = pd.read_csv(valid_csv_path)
    
    metadata_columns = ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']
    path_column = 'Path'
    label_columns = [c for c in df_train.columns if c not in (metadata_columns + [path_column])]
    
    print(metadata_columns, len(metadata_columns))
    print(path_column, len(path_column))
    print(label_columns, len(label_columns))
    
    return 
    
    ## metadata_columns : Sex, Age, Frontal/Lateral, AP/PA
    categoric_columns = ['Sex', 'Frontal/Lateral', 'AP/PA']
    print(df_train.iloc[0:10][categoric_columns])    
    for c in categoric_columns:
        encode_categoric_column(df_train, c, data_dict)
        df_train[c] = df_train[c].replace(data_dict[c]['label2id'])
        
    ## there is no Path or Age with nan value both in train and valid dataframes !!!
    print(df_train.iloc[0:10][categoric_columns])    
    
    df_train['Path'] = fix_df_path(df_train, path)
    df_valid['Path'] = fix_df_path(df_valid, path)
    
    df_train = treat_df_label_columns(df_train, label_columns, inplace=True)
    df_valid = treat_df_label_columns(df_valid, label_columns, inplace=True)
    
    