import torch
import torch.nn as nn 

# Initial configuration
CONFIG = {
    'Data_Path': 'Datasets/ClothesModified/',  #Clothes #Data_Logo2K #Dataset(1and2)
    'Save_To': 'models',
    'Cross_Validation': True,
    'k_folds': 5,
    'Split_Size': 0.8, 
    'Models': ['Resnet50'], #, 'Resnet101', 'EfficientNetB7'
    'Fine_Tune': False,
    'Pretrained': True,
    'Num_Epochs': 1, 
    'Batch_size': 32, 
    'Optimizer': {
       'Name': 'Adam',
        'lr': 0.001,
        'weight_decay':0,
        'momentum': 0.9
    },
    'Loss_Function': 'CrossEntropyLoss',
    'label_smoothing': 0.1,
    'Scheduler': {
       'Name': 'StepLR', 
        #'Name': 'StepLR', CustomLRScheduler, ReduceLROnPlateau, CosineAnnealingLR, #CustomLRSchedulerModified
        'step_size': 10, 
        'gamma': 0.1, 
        'Warmup_Epochs': 150, 
        'Stop_LR': 10, 
        'Base_LR': 1e-3, 
        'Final_LR': 1e-5
    },
    'EarlyStopping': {
        'Patience': 20, 
        'min_delta': 0, 
        'verbose': True
    }
}

CLASSIFIER = None
    
def set_classifier_layer(num_features: int, num_classes: int):
    global CLASSIFIER
    CLASSIFIER =  nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
            nn.Linear(num_features, 1024),
            #nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, num_classes)
            #nn.ReLU(),
            #nn.BatchNorm1d(512),
            #nn.Dropout(p=0.3, inplace=False),
            #nn.Linear(512, num_classes),
            #nn.ReLU(),
           # nn.BatchNorm1d(num_classes),
            #nn.Dropout(p=0.3, inplace=True)
        )
    
def get_classifier_layer():
    global CLASSIFIER
    return CLASSIFIER

def set_config(config: dict):
    global CONFIG  
    CONFIG = config

def get_config() -> dict:
    global CONFIG
    return CONFIG

    
#resnet18 =  self.classifier_layer = nn.Sequential(
			#nn.Linear(128, num_classes)
            #nn.Linear(num_features, num_classes),
            #nn.BatchNorm1d(num_classes),
            #nn.Dropout(p=0.5, inplace=True)
            #nn.Linear(128, num_classes)
            # nn.LogSoftmax(dim=1))

#efficientnet_b0 = nn.Sequential(
#            nn.Dropout(p=0.3, inplace=False),
#            nn.Linear(num_features, num_classes),)