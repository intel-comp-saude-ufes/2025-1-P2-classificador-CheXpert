from torch import nn
import torch
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lib.early_stopping import EarlyStopping
from lib.history import History

def save_checkpoint(path, model, optimizer, epoch, **kwargs):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)

def file_exists(path):
    '''
    Verifica se um arquivo existe.
    '''
    from pathlib import Path
    p = Path(path)
    return (p.exists() and p.is_file())


def eval(model:torch.nn.Module, eval_dataset:torch.utils.data.Dataset, device:torch.device):
    '''
    Avalia o modelo em um dataset e retorna listas contendo os labels reais e as predições (y_true e y_pred).
    '''
    import torch
    from torch.utils.data.dataloader import DataLoader
    y_true = []
    y_pred = []
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=True)
    model.eval()
    with torch.no_grad():
        for X,y in eval_dataloader:
            X,y = X.to(device), y.to(device)        
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())    
    return (y_true, y_pred)


def one_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer, loss_fn, device: torch.device, is_eval:bool=False, metric_obj = None):
    '''
    Executa uma época, se is_eval=True, não calcula os gradientes e coloca o modelo em modo de avaliação.
    '''
    loss_acc = 0.0

    if is_eval:
        model.eval()
        context = torch.no_grad()
    else:
        model.train()
        context = torch.enable_grad()

    with context:
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
                    
            if metric_obj is not None:
                metric_obj.update(logits, y)
            
            loss = loss_fn(logits, y)
            loss_acc += loss.item()

            if not is_eval:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    return loss_acc


def construct_paths(save_path, save_name):
    if save_path is not None:    
        model_path = os.path.join(save_path, save_name)
        best_model_path = os.path.join(save_path, f'best_{save_name}')
    return model_path, best_model_path


def att_desc_tqdm_bar(tqdm_bar, epoch, train_loss, train_accuracy, val_loss, min_val_loss, val_accuracy, new_best=False):
    desc_str = (f"Epoch {epoch+1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2f}% | "
                f"Val Loss/ Min: {val_loss:.4f}/{min_val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.2f}% | "
                f"New Best? {new_best}"
                )
    tqdm_bar.set_description(desc=desc_str)
    tqdm_bar.refresh()
    return


def train(model: nn.Module, train_dataloader: Dataset, val_dataloader :Dataset, loss_fn, optimizer,   
        epochs=30, verbose= False, save_path: str|None=None, save_name='model.pt', warmup=5, patience=10, device=torch.device('cpu'), metrics=None):
    '''
    Executa um treinamento completo do modelo passado, utiliza a função CrossEntropyLoss e otimizador Adam.
    Early Stopping com paciência de 10 épocas e Warmup definido em 5 épocas.
    O Early Stopping é feito em cima do valor obtido da Loss média do Batch no conjunto de validação da época.
    '''
    
    model_path, best_model_path = construct_paths(save_path, save_name)
    
    no_improve_counter = 0
    new_best = False
    min_val_loss_mean = float('inf')
    last_epoch = 0
    starting_epoch = 0
    
    early_stopper = EarlyStopping(patience=patience, mode='min', monitor='val_loss')
    history = History()
    
    ## loading checkpoint if the path exists
    if (model_path is not None) and file_exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False, map_location=device) ## weights_only=False because the 'True' value is now the default one.
    
        ## model
        model_state = checkpoint['model_state_dict']
        model.load_state_dict(model_state)
        
        ## optimizer
        optim_state = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optim_state)
        
        ## epoch
        starting_epoch = checkpoint['epoch'] + 1 
        
        ## metadata
        metadata = checkpoint['metadata']
        
        min_val_loss_mean = metadata['min_val_loss']
        history.set_inner_dict(metadata['history'])
        new_best = metadata['is_best']
                
    if verbose:
        print((f'Train Config:'
               f'\nEpochs: {epochs}'
               f'\nDevice: {device}'
               f'\nModel Arch: {model.__class__.__name__}'
               f'\nCriterion : {loss_fn.__class__.__name__}'
               f'\nOptimizer : {optimizer.__class__.__name__}'
        ))
        
    ## tqdm progress bar
    epoch_tqdm_bar = tqdm(range(starting_epoch, epochs))
    
    if starting_epoch != 0:
        att_desc_tqdm_bar(epoch_tqdm_bar, starting_epoch-1, history.last('train_loss'), history.last('train_accuracy'),
                          history.last('val_loss'), min_val_loss_mean, history.last('val_accuracy'), new_best=new_best)
        
    else : epoch_tqdm_bar.set_description(desc=f'No Info')
    
    for epoch in epoch_tqdm_bar:
        last_epoch = epoch
        
        ### training
        train_loss = one_epoch(model, train_dataloader, optimizer, loss_fn, device, metric_obj=metrics)
        train_loss_mean = train_loss / len(train_dataloader)

        history.log(train_loss=train_loss_mean)
        train_metrics = metrics.compute()
        metrics.reset()
        
        history.log(**{f'train_{k}': v for k,v in train_metrics.items()})
        
        ## validation
        val_loss = one_epoch(model, val_dataloader, optimizer, loss_fn, device, is_eval=True, metric_obj=metrics)
        val_loss_mean = val_loss / len(val_dataloader)     
    
        history.log(val_loss=val_loss_mean)
        val_metrics = metrics.compute()
        metrics.reset()
        
        history.log(**{f'val_{k}': v for k,v in val_metrics.items()})
        
        should_stop = False
        new_best=False
        
        if epoch >= warmup:
                should_stop = early_stopper.step(val_metrics)
                new_best = early_stopper.is_best()
        else:
            new_best = val_loss_mean < min_val_loss_mean
        
        if new_best:
            min_val_loss_mean = val_loss_mean
            if best_model_path:
                metadata = {
                    'min_val_loss' : min_val_loss_mean,
                    'history' : history.get_inner_dict(),
                    'is_best' : new_best,
                }
                save_checkpoint(best_model_path, model, optimizer, epoch,
                                metadata=metadata)
        
        
        att_desc_tqdm_bar(epoch_tqdm_bar, epoch, history.last('train_loss'), history.last('train_accuracy'), 
                          history.last('val_loss'), min_val_loss_mean, history.last('val_accuracy'), new_best=new_best)
        
        ## early stopping
        if should_stop: break
        
        if model_path is not None:
                if ((epoch + 1) % 5 == 0) or new_best:
                        metadata = {
                            'min_val_loss' : min_val_loss_mean,
                            'history' : history.get_inner_dict(),
                            'is_best' : new_best,
                        }
                        save_checkpoint(model_path, model, optimizer, epoch,
                                        metadata=metadata)
                                              
    if no_improve_counter >= patience:
        if verbose: 
            _msg_str = f'At Epoch [{epoch + 1}], it had {patience} iterations with no improvement on the validation dataset. Stopping ...' 
            print(_msg_str)           
        
    return history, model