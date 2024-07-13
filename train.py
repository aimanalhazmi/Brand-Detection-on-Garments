import os
import json
from pathlib import Path
import time
import torch
import torch.optim as optim
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import numpy as np
import random
import data_setup
import engine
from engine import CustomLRScheduler, EarlyStopping, CustomLRScheduler_modified
import model_builder
from config import get_config, get_classifier_layer
from utilities.helper import save_model, create_target_dir, create_dir, log_stdout_to_file, set_seed
from utilities import visual
from sklearn.metrics import accuracy_score
import sys
import warnings

warnings.filterwarnings("ignore")


def set_optimizer(model: torch.nn.Module, optimizer: dict):
    opt = optimizer["Name"]
    if opt == 'SGD':
        print(f'[INFO]: Setting Optimizer: {opt}..')
        return optim.SGD(params=model.parameters(), lr=optimizer["lr"], momentum=optimizer["momentum"], weight_decay=optimizer["weight_decay"])
    elif opt == 'Adam':
        print(f'[INFO]: Setting Optimizer: {opt}..')
        return optim.Adam(model.parameters(), lr=optimizer["lr"], weight_decay=optimizer["weight_decay"])
    elif opt == 'AdamW':
        print(f'[INFO]: Setting Optimizer: {opt}..')
        return optim.AdamW(model.parameters(), lr=optimizer["lr"], weight_decay=optimizer["weight_decay"])    
    else:
        print(f'[INFO]: Setting default Optimizer: Adam..')
        return optim.Adam(model.parameters(), lr=optimizer["lr"], weight_decay=optimizer["weight_decay"])


def set_loss_function(label_smoothing: float, loss_fn: str):
    if loss_fn == 'CrossEntropyLoss':
        print(f'[INFO]: Setting Loss Function: {loss_fn}..')
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        print(f'[INFO]: Setting default Loss Function:: CrossEntropyLoss..')
        return torch.nn.CrossEntropyLoss()


def set_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    scheduler = config['Scheduler']
    if scheduler:
        print(f'[INFO]: Setting Scheduler: {scheduler["Name"]}..')
        if scheduler['Name'] == 'StepLR':
            return StepLR(optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma'])
        
        elif scheduler['Name'] == 'CustomLRScheduler':
            return CustomLRScheduler(optimizer, warmup_epochs=scheduler['Warmup_Epochs'], stop_lr=scheduler['Stop_LR'], total_epochs=config['Num_Epochs'], base_lr=scheduler['Base_LR'], final_lr=scheduler['Final_LR'])
        
        elif scheduler['Name'] == 'CustomLRSchedulerModified':
            return CustomLRScheduler_modified(optimizer, warmup_epochs=scheduler['Warmup_Epochs'], total_epochs=config['Num_Epochs'], base_lr=scheduler['Base_LR'], final_lr=scheduler['Final_LR'])
        
        elif scheduler['Name'] == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, 'min')
        
        elif scheduler['Name'] == 'CosineAnnealingLR':
            return CosineAnnealingLR(optimizer, T_max=scheduler['Warmup_Epochs'])
        
    print(f'[INFO]: No Scheduler used..')
    return None


def print_config(config: dict, model: torch.nn.Module, target_dir: str, device: torch.device):
    with open(f'{target_dir}/config.json', 'w') as file:
        json.dump(config, file)
    print(f'\nConfiguration Settings: {json.dumps(config, indent=4)}')
    # Summary
    name = model.__class__.__name__
    if name=='Vgg16':
        print(f"{summary(model)}\n")
    else:
        print(f"{summary(model, input_size=(32, 3, 224, 224))}\n")
    print(f'Classifier_layer: {model.classifier}\n')


def set_EarlyStopping(earlyStopping: dict, target_dir: str):
    checkpoint_path = create_dir(path=target_dir, dir_name="Checkpoint")
    return EarlyStopping(checkpoint_path, patience=earlyStopping['Patience'], min_delta=earlyStopping['min_delta'], verbose=earlyStopping['verbose'])
    

def execute_with_cross_validation(model_name, config, target_dir):
    checkpoint_path = create_dir(path=target_dir, dir_name="Checkpoint")

    config['Save_To'] = str(target_dir)
    with log_stdout_to_file(f'{target_dir}/experiment_log.txt'):
        print('[INFO]: Cross Validation is applied..')
        set_seed(42)
        # Load the training and validation datasets.
        dataset, class_names = data_setup.get_dataset(config['Data_Path'])
        #train_dataloader, valid_dataloader = data_setup.get_dataloaders(train_dataset, valid_dataset, config['Batch_size'])
        if config['k_folds']:
            k_folds = config['k_folds']
        else:
            k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        data_setup.print_dataset_info(dataset, kf)
        # Visual Data Before After Transformation...
        
        transforms_train, transforms_valid  = data_setup.get_transforms()
        visual.plot_img_before_after_transformation(dataset, transforms_train, class_names, target_dir)
        
        # Setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    results_all = []#
    set_seed(42)
    for fold, (train_ids, valid_ids) in enumerate(kf.split(np.arange(len(dataset)))):
        with log_stdout_to_file(f'{target_dir}/experiment_log.txt'):
            model = model_builder.create_model(model_name, config=config, num_classes=len(class_names)).to(device)
            optimizer = set_optimizer(model=model, optimizer=config['Optimizer'])
            scheduler = set_scheduler(config=config, optimizer=optimizer)
            loss_fn = set_loss_function(label_smoothing=config['label_smoothing'], loss_fn=config['Loss_Function'])
            early_stopping = set_EarlyStopping(earlyStopping=config['EarlyStopping'], target_dir=target_dir)
            target_dir_fold = create_dir(path=target_dir, dir_name=f"{fold+1}")
            train_dataloader, valid_dataloader = data_setup.get_dataloaders_for_cross_validation(dataset, train_ids, valid_ids, batch_size=config['Batch_size'], num_workers=4)
            print_config(config=config, model=model, target_dir=target_dir, device=device)
            print(f'----------FOLD {fold+1}/{k_folds}----------')
        start_time = time.time()
        results, model = engine.train_valid(model=model,
                                            train_dataloader=train_dataloader,
                                            valid_dataloader=valid_dataloader,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            early_stopping=early_stopping,
                                            target_dir=target_dir_fold,
                                            epochs=config['Num_Epochs'], device=device,
                                            loss_fn=loss_fn)

        save_model(model=model, target_dir=target_dir_fold, model_name='last_model.pth')
        to_add = {'results':results, 'model': model, 'target_dir':target_dir_fold, 'valid_dataloader': valid_dataloader}
        results_all.append(to_add)
        early_stopping.counter = 0
    time_elapsed = time.time() - start_time
    with log_stdout_to_file(f'{target_dir}/experiment_log.txt'):
        print('TRAINING COMPLETE IN {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Validation Acc: {:4f}%\n'.format(results['best_accuracy']))
        for fold, item in enumerate(results_all):
            print(f'[INFO]: Strating Evaluation for Fold: {fold+1}')
            evaluate(item['results'], item['target_dir'], item['model'], item['valid_dataloader'], loss_fn, device, class_names)
    return results, model, target_dir


def execute(model_name, config, target_dir):
    checkpoint_path = create_dir(path=target_dir, dir_name="Checkpoint")

    config['Save_To'] = str(target_dir)
    with log_stdout_to_file(f'{target_dir}/experiment_log.txt'):
        # Load the training and validation datasets.
        train_dataset, valid_dataset, class_names = data_setup.get_datasets(config['Data_Path'], config['Split_Size'])
        train_dataloader, valid_dataloader = data_setup.get_dataloaders(train_dataset, valid_dataset, config['Batch_size'],  augmented=True, num_workers=4)
        print(f"[INFO]: Number of Training Images: {len(train_dataset)}")
        print(f"[INFO]: Number of Validation Images: {len(valid_dataset)}")
        print(f"[INFO]: Number of Classes: {len(class_names)}\n")
        # Visual Data Before After Transformation...
        transforms_train, transforms_valid  = data_setup.get_transforms()
        visual.plot_img_before_after_transformation(train_dataset, transforms_train, class_names, target_dir)
        # Setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        set_seed(42)
        model = model_builder.create_model(model_name, config=config, num_classes=len(class_names)).to(device)
        # Define the loss function and optimizer
        loss_fn = set_loss_function(label_smoothing=config['label_smoothing'], loss_fn=config['Loss_Function'])
        optimizer = set_optimizer(model=model, optimizer=config['Optimizer'])
        early_stopping = set_EarlyStopping(earlyStopping=config['EarlyStopping'], target_dir=target_dir)
        scheduler = set_scheduler(config=config, optimizer=optimizer)

        print_config(config=config, model=model, target_dir=target_dir, device=device)

    start_time = time.time()
    results, model = engine.train_valid(model=model,
                                        train_dataloader=train_dataloader,
                                        valid_dataloader=valid_dataloader,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        early_stopping=early_stopping,
                                        target_dir=target_dir,
                                        epochs=config['Num_Epochs'], device=device,
                                        loss_fn=loss_fn)
    save_model(model=model, target_dir=target_dir, model_name='last_model.pth')
    time_elapsed = time.time() - start_time
    with log_stdout_to_file(f'{target_dir}/experiment_log.txt'):
        print('TRAINING COMPLETE IN {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Validation Acc: {:4f}%\n'.format(results['best_accuracy']))
        evaluate(results, target_dir, model, valid_dataloader, loss_fn, device, class_names)
    return {'results':results, 'model':model, 'target_dir':target_dir}


def evaluate(results, target_dir, model, valid_dataloader, loss_fn, device, class_names):
    title = 'Performance'
    visual.plot_results(results, target_dir, title)
    print('-' * 80)
    print('\nEvaluation...\n')
    y_pred = []
    y_true = []
    target_dir = str(target_dir)
    # Perform validation using the engine's valid_model method
    history = engine.valid_model(model, valid_dataloader, loss_fn, device)

    # Print validation results
    print('[INFO] Validation Loss: {:.4f} | Acc: {:.4f}% | Weighted F1: {:.4f}'.format(history[0], history[1], history[4]))

    # Append results to lists
    y_pred.extend(history[2])
    y_true.extend(history[3])

    # Compute and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy on Validation set using accuracy_score from sklearn: {accuracy*100}%")

    visual.get_classification_report(y_true=y_true, y_pred=y_pred, target_dir=target_dir)
    # Plot confusion matrix
    #visual.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_names=class_names, target_dir=target_dir)
    visual.plot_lr(learning_rates=results['learning_rates'], target_dir=target_dir)


def train_valid_evaluate():
    config = get_config()
    models = config['Models']
    target_dir = create_target_dir(target_dir=config['Save_To'])
    for model in models:
        print(f'Strating to train {model}...')
        model_dir = create_dir(path=target_dir, dir_name=model)
        if config['Cross_Validation']:
            results = execute_with_cross_validation(model_name=model, config=config, target_dir=model_dir)
        else:
            results = execute(model_name=model, config=config, target_dir=model_dir)
        print('-' * 80)

    print('All models have been trained.')


if __name__ == "__main__":
    train_valid_evaluate()