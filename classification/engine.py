import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from time import sleep
from sklearn.metrics import confusion_matrix, f1_score
from utilities.helper import save_model, get_num_correct
import math

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        # Check if validation loss improved
        if val_loss < self.val_loss_min - self.min_delta:
            self.counter = 0
            self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss
        else:
            self.counter += 1
            if self.verbose:
                print(f'[INFO] EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:

                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'[INFO] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model=model, target_dir=self.checkpoint_path, model_name=f'checkpoint_{self.counter}.pth')


class CustomLRScheduler:
    def __init__(self, optimizer, warmup_epochs, stop_lr, total_epochs, base_lr, final_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.stop_lr = stop_lr
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.current_epoch = 0

    def step(self, epoch):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch / self.warmup_epochs)
        elif epoch < self.total_epochs - self.stop_lr:
            lr = self.final_lr + (self.base_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.total_epochs - self.stop_lr - self.warmup_epochs)))
        else:
            lr = self.final_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]

class CustomLRScheduler_modified:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, final_lr):
        self.optimizer = optimizer
        self.change_epoch = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.current_epoch = 0

    def step(self, epoch):
        self.current_epoch = epoch
        if epoch < self.change_epoch:
            lr = self.base_lr
        else:
            # Linear decrease from base_lr to final_lr
            lr = self.base_lr - (self.base_lr - self.final_lr) * (epoch - self.change_epoch) / (self.total_epochs - self.change_epoch)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]
    
def train_model(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler,
                device: torch.device):
    model.to(device)
    # training mode
    model.train()
    print('Training...')
    train_loss, train_corrects = 0.0, 0
    total_samples = 0
    y_pred = []
    y_true = []
    # loop through the training batches..
    for batch, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # forward inputs and get output
        outputs = model(inputs)

        # Calculate loss (per batch)
        loss = loss_fn(outputs, labels)
        train_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)


        loss.backward()

        optimizer.step()

        # Calculate Accuracy
        _, preds = torch.max(outputs.data, 1)
        train_corrects += get_num_correct(preds, labels)
        # Extend
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

        
    avg_train_loss = train_loss / total_samples
    avg_train_acc = train_corrects / total_samples * 100
    weighted_f1 = f1_score(y_true, y_pred, average='weighted') * 100

    # Print progress
    return avg_train_loss, avg_train_acc, weighted_f1


def valid_model(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                device: torch.device):
    model.to(device)
    model.eval()
    print('Validation...')
    val_loss = 0.0
    val_corrects = 0
    total_samples  = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)
            except AttributeError as e:
                print(f"Error processing batch {batch}: {e}")
                continue
            outputs = model(inputs)

            # Calculate loss (per batch)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Calculate Accuracy
            _, preds = torch.max(outputs.data, 1)
            val_corrects += get_num_correct(preds, labels)

            # Extend
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    avg_valid_loss = val_loss / total_samples
    avg_valid_acc = val_corrects / total_samples * 100

    # Calculate the weighted F1 score
    weighted_f1 = f1_score(y_true, y_pred, average='weighted') * 100

    return [avg_valid_loss, avg_valid_acc, y_pred, y_true, weighted_f1]


def schedulerStep(scheduler, optimizer, epoch, valid_loss):
    lr = optimizer.param_groups[0]['lr']
    if scheduler:
        if isinstance(scheduler, (CustomLRScheduler, CustomLRScheduler_modified)):
            scheduler.step(epoch+1)
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_loss)
        else:
            scheduler.step()
        if hasattr(scheduler, 'get_last_lr'):
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']
        print(f"[INFO] LR: {lr}")
    return lr

def train_valid(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                valid_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler,
                early_stopping: EarlyStopping,
                target_dir: str,
                epochs: int, device: torch.device,
                loss_fn: torch.nn.Module = nn.CrossEntropyLoss()):

    model.to(device)
    results = {'train_loss': [], 'train_acc': [],
               'valid_loss': [], 'valid_acc': [],
               'y_pred': [], 'y_true': [],
               'train_weighted_f1': [], 'valid_weighted_f1': [], 'best_accuracy': 0.0,
               'best_f1_score': 0.0, 'learning_rates':[]
               }

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")

        """ Training Phase """
        avg_train_loss, avg_train_acc, weighted_f1 = train_model(model, train_dataloader, loss_fn, optimizer, scheduler, device)
        # Append result
        results['train_loss'].append(avg_train_loss)
        results['train_acc'].append(avg_train_acc)
        results['train_weighted_f1'].append(weighted_f1)

        """ Validation Phase """
        history = valid_model(model, valid_dataloader, loss_fn, device)
        # Append result
        results['valid_loss'].append(history[0])
        results['valid_acc'].append(history[1])
        results['valid_weighted_f1'].append(history[4])
        results['y_pred'].extend(history[2])
        results['y_true'].extend(history[3])
        
        lr = schedulerStep(scheduler, optimizer, epoch, history[0])
        results['learning_rates'].append(lr)
        print(f"[INFO] LR: {lr}")
        print('[INFO] Training Loss: {:.4f} | Acc: {:.4f}% | Weighted F1: {:.4f}'.format(avg_train_loss, avg_train_acc, weighted_f1))
        print('[INFO] Validation Loss: {:.4f} | Acc: {:.4f}% | Weighted F1: {:.4f}'.format(history[0], history[1], history[4]))

        early_stopping(history[0], model)
        if early_stopping.early_stop:
            print("Early stopping....")
            print('-' * 80)
            sleep(3)
            break
        if history[1] > results['best_accuracy']:
            results['best_accuracy'] = history[1]
            results['best_f1_score'] = history[4]

            print('[INFO] Improvement-Detected, Best model updated')
            save_model(model=model, target_dir=target_dir, model_name='best_model.pth')
        print('-' * 80)
        sleep(3)
    
    return results, model
