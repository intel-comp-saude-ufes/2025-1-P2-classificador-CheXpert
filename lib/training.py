from torch import nn
import torch
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchmetrics.classification import MultilabelF1Score


def save_my_checkpoint(model:torch.nn.Module, optimizer, loss_fn, min_val_loss:float, epoch : int, history: dict, path : str, is_best:bool=False):
    '''
    Função auxiliar que monta o dicionário do checkpoint guardando todos os atributos passados e salva em path.
    '''
    
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
        
    torch.save({
        'model_class' : str(model.__class__.__name__),
        'optimizer_class' : str(optimizer.__class__.__name__),
        'loss_class' : str(loss_fn.__class__.__name__),
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss_fn' : loss_fn,
        'epoch' : epoch,
        'min_val_loss_mean' : min_val_loss,
        'history' : history,
        'is_best': is_best,
    }, path)
    
    return


def file_exists(path):
    '''
    Verifica se um arquivo existe.
    '''
    from pathlib import Path
    p = Path(path)
    return (p.exists() and p.is_file())


def calculate_metrics(y_true : list, y_pred: list):
    '''
    Retorna as métricas de acurácia e f1-score para as entradas.
    '''
    from sklearn.metrics import accuracy_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    return (accuracy, f1_macro)


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


def one_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer, loss_fn, device: torch.device, is_eval:bool=False):
    '''
    Executa uma época, se is_eval=True, não calcula os gradientes e coloca o modelo em modo de avaliação.
    '''
    #y_true = []
    #y_pred = []
    f1_metric_obj = MultilabelF1Score(num_labels=14, average="micro").to('cpu')
    loss_acc = 0.0

    if is_eval:
        model.eval()
        context = torch.no_grad()
    else:
        model.train()
        context = torch.enable_grad()
    i = 0
    with context:
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            #preds = torch.argmax(logits, dim=1)
            #y_pred.extend(preds.cpu().numpy())
            #y_true.extend(y.cpu().numpy())
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            f1_metric_obj.update(preds, y)
            f1_score = f1_metric_obj.compute().item()

            loss = loss_fn(logits, y)
            loss_acc += loss.item()

            if not is_eval:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            i+=1
            if (i % 10) == 0:
                print(f'batch {i} done, f1_score : {f1_score}')
    
    f1_score = f1_metric_obj.compute().item()
    return f1_score, loss_acc

def construct_paths(save_path, save_name):
    if save_path is not None:    
        model_path = os.path.join(save_path, save_name)
        best_model_path = os.path.join(save_path, f'best_{save_name}')
    return model_path, best_model_path
    

def train(model: nn.Module, train_dataset: Dataset, val_dataset :Dataset, loss_fn, optimizer,   
        epochs=30, verbose= False, save_path: str|None=None, save_name='model.pt', warmup=5, patience=10, device=torch.device('cpu')):
    '''
    Executa um treinamento completo do modelo passado, utiliza a função CrossEntropyLoss e otimizador Adam.
    Early Stopping com paciência de 10 épocas e Warmup definido em 5 épocas.
    O Early Stopping é feito em cima do valor obtido da Loss média do Batch no conjunto de validação da época.
    '''
    
    model_path, best_model_path = construct_paths(save_path, save_name)
    
    num_workers = max(1,os.cpu_count()-1)

    
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    
    model = model.to(device)
    #model = model.to(device)
    #loss_fn = nn.CrossEntropyLoss(reduction='mean')
    #optimizer = torch.optim.Adam(params= model.parameters(), lr = 0.001)
    
           
    history = {
        'train_loss' : [],
        'val_loss' : [],
        'train_accuracy' : [],
        'val_accuracy' : [],
        'train_f1_macro' : [],
        'val_f1_macro' : [],
    }
    
    warmup_threshold = warmup
    patience_threshold = patience
    no_improve_counter = 0
    new_best_flag = False
    min_val_loss_mean = float(1e6)
    last_epoch = 0
    starting_epoch = 0
    
    ## loading checkpoint if the path exists
    if (model_path is not None) and file_exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False, map_location=device) ## weights_only=False because the 'True' value is now the default one.
    
        model_state = checkpoint['model_state_dict']
        model.load_state_dict(model_state)
        
        optim_state = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optim_state)
        
        loss_fn = checkpoint['loss_fn']
        starting_epoch = checkpoint['epoch'] + 1 
        min_val_loss_mean = checkpoint['min_val_loss_mean']
        
        history = checkpoint['history']
        new_best_flag = checkpoint['is_best']
        
        
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
        att_desc_tqdm_bar(epoch_tqdm_bar, starting_epoch-1, history['train_loss'][-1], history['train_accuracy'][-1],
                          history['val_loss'][-1], min_val_loss_mean, history['val_accuracy'][-1], new_best=new_best_flag)
        
    else : epoch_tqdm_bar.set_description(desc=f'No Info')

    
    for epoch in epoch_tqdm_bar:
        last_epoch = epoch
        
        ### training
        train_f1_score, train_loss = one_epoch(model, train_dataloader, optimizer, loss_fn, device)
        train_loss_mean = train_loss / len(train_dataloader)
        #train_accuracy, train_f1_macro = calculate_metrics(y_true, y_pred)
        
        history['train_loss'].append(train_loss_mean)
        history['train_accuracy'].append(0)
        history['train_f1_macro'].append(0)
        
        ### validation
        val_f1_score, val_loss = one_epoch(model, val_dataloader, optimizer, loss_fn, device, is_eval=True)
        val_loss_mean = val_loss / len(val_dataloader)     
        #val_accuracy, val_f1_macro = calculate_metrics(y_true, y_pred)
        
        history['val_loss'].append(val_loss_mean)
        history['val_accuracy'].append(0)
        history['val_f1_macro'].append(0)
        
        ## warmup logic
        ## if the warmup was done and had no improvement
        new_best_flag = False
        if (val_loss_mean >= min_val_loss_mean): ## no improvement
            if epoch >= warmup_threshold: ## warmup done
                no_improve_counter = no_improve_counter + 1 ## att counter
        else: ## new min, reset counter, save best model
            new_best_flag = True
            no_improve_counter = 0
            min_val_loss_mean = val_loss_mean
            if save_path is not None:
                save_my_checkpoint(model, optimizer, loss_fn, min_val_loss_mean, epoch, history, best_model_path,is_best=True)
        
        att_desc_tqdm_bar(epoch_tqdm_bar, epoch, train_loss, train_f1_score, 
                          val_loss, min_val_loss_mean, val_f1_score, new_best=new_best_flag)
        
        ## early stopping
        if no_improve_counter >= patience_threshold:           
            break
        
        if ((epoch+1)%5==0) or (new_best_flag): ## saving model in from 5 to 5 iterations or if the new best is found
            save_my_checkpoint(model, optimizer, loss_fn, min_val_loss_mean, epoch, history, model_path, is_best=new_best_flag)
                      
    if no_improve_counter >= patience_threshold:
        if verbose: 
            _msg_str = f'At Epoch [{epoch + 1}], it had {patience_threshold} iterations with no improvement on the validation dataset. Stopping ...' 
            print(_msg_str)           
        
        
    return history, model