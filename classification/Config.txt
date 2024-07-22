CONFIG = {
    'Data_Path': 'Datasets/ClothesModified/',  #Clothes #Data_Logo2K #Dataset(1and2)
    'Save_To': 'models',
    'Cross_Validation': False,
    'k_folds': 5,
    'Split_Size': 0.8, 
    'Models': ['Resnet50', 'EfficientNetB7', 'Vgg16', 'Resnet101' ], #, 'Resnet101', 'EfficientNetB7'
    'Fine_Tune': False,
    'Pretrained': True,
    'Num_Epochs': 300, 
    'Batch_size': 32, 
    'Optimizer': {
       'Name': 'Adam',
        'lr': 0.001,
         'weight_decay':0.001,
        'momentum': 0.9
    },
    'Loss_Function': 'CrossEntropyLoss',
    'label_smoothing': 0.1,
    'Scheduler': {
       'Name': 'CustomLRSchedulerModified', 
        #'Name': 'StepLR', CustomLRScheduler, ReduceLROnPlateau, CosineAnnealingLR, #CustomLRSchedulerModified
        'step_size': 10, 
        'gamma': 0.1, 
        'Warmup_Epochs': 80, 
        'Stop_LR': 150, 
        'Base_LR': 1e-3, 
        'Final_LR': 1e-5
    },
    'EarlyStopping': {
        'Patience': 20, 
        'min_delta': 0, 
        'verbose': True
    }
}