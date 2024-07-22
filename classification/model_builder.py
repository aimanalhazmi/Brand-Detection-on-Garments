import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ExponentialLR
from config import set_classifier_layer, get_classifier_layer


class Resnet18(nn.Module):
    def __init__(self,classifier, pretrained, fine_tune):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
                # Remove the original fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove the last layer
        
        self.classifier = classifier
        model_name = self.__class__.__name__
        print(f"Model: {model_name}")
        print(f'[INFO]: Number of infeatures:{num_features}')
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            self.freeze_layers()
        
    def forward(self, x):
        # Extract features from the base model
        x = self.model(x)  # Here, x will be the output from the average pooling layer
        # Flatten the features
        x = torch.flatten(x, 1)
        # Pass features through the new classifier layers
        x = self.classifier(x)
        return x
    
    def freeze_layers(self):
        # Freeze layers except for 'layer3', 'layer4', and 'classifier_layer'
        for name, child in self.model.named_children():
            if name in ['layer3', 'layer4']:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False 
            
        for params in self.classifier.parameters():
            params.requires_grad = True   



class Resnet50(nn.Module):
    def __init__(self, classifier, pretrained, fine_tune):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
                # Remove the original fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove the last layer
        
        self.classifier = classifier

        model_name = self.__class__.__name__
        print(f"Model: {model_name}")
        print(f'[INFO]: Number of infeatures:{num_features}')
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            self.freeze_layers()
        
    def forward(self, x):
        # Extract features from the base model
        x = self.model(x)  # Here, x will be the output from the average pooling layer
        # Flatten the features
        x = torch.flatten(x, 1)
        # Pass features through the new classifier layers
        x = self.classifier(x)
        return x
    
    def freeze_layers(self):
        # Freeze layers except for 'layer3', 'layer4', and 'classifier_layer'
        for name, child in self.model.named_children():
            if name in ['layer3', 'layer4']:
                for param in child.parameters():
                    param.requires_grad = True 
            else:
                for param in child.parameters():
                    param.requires_grad = False 
            
        for params in self.classifier.parameters():
            params.requires_grad = True 
            

class Resnet101(nn.Module):
    def __init__(self, classifier, pretrained, fine_tune):
        super(Resnet101, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
                # Remove the original fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove the last layer
        
        self.classifier = classifier

        model_name = self.__class__.__name__
        print(f"Model: {model_name}")
        print(f'[INFO]: Number of infeatures:{num_features}')
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            self.freeze_layers()
        
    def forward(self, x):
        # Extract features from the base model
        x = self.model(x)  # Here, x will be the output from the average pooling layer
        # Flatten the features
        x = torch.flatten(x, 1)
        # Pass features through the new classifier layers
        x = self.classifier(x)
        return x
    
    def freeze_layers(self):
        # Freeze layers except for 'layer3', 'layer4', and 'classifier_layer'
        for name, child in self.model.named_children():
            if name in ['layer3', 'layer4']:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False 
            
        for params in self.classifier.parameters():
            params.requires_grad = True 
                
class EfficientNetB0(nn.Module):
    def __init__(self, classifier, pretrained=True, fine_tune=True):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)
        # Remove the original fully connected layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Identity()  # Remove the last layer

        # Define the new classifier layer
        self.classifier = classifier
        print(self.model)

        model_name = self.__class__.__name__
        print(f"Model: {model_name}")
        print(f'[INFO]: Number of in_features: {num_features}')
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        else:
            print('[INFO]: Freezing hidden layers...')
            self.freeze_layers()
        
    def forward(self, x):
        # Extract features from the base model
        x = self.model(x)  # Here, x will be the output from the average pooling layer
        # Flatten the features
        x = torch.flatten(x, 1)
        # Pass features through the new classifier layers
        x = self.classifier(x)
        return x
    
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze specific layers (features[7] and features[8])
        for i in [7, 8]:
            for param in self.model.features[i].parameters():
                param.requires_grad = True

        # Ensure classifier layers are unfrozen
        for param in self.classifier.parameters():
            param.requires_grad = True

class EfficientNetB7(nn.Module):
    def __init__(self, classifier, pretrained=True, fine_tune=True):
        super(EfficientNetB7, self).__init__()
        self.model = models.efficientnet_b7(pretrained=pretrained)
        # Remove the original fully connected layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Identity()  # Remove the last layer

        # Define the new classifier layer
        self.classifier = classifier

        model_name = self.__class__.__name__
        print(f"Model: {model_name}")
        print(f'[INFO]: Number of in_features: {num_features}')
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        else:
            print('[INFO]: Freezing hidden layers...')
            self.freeze_layers()
        
    def forward(self, x):
        # Extract features from the base model
        x = self.model(x)  # Here, x will be the output from the average pooling layer
        # Flatten the features
        x = torch.flatten(x, 1)
        # Pass features through the new classifier layers
        x = self.classifier(x)
        return x
    
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze specific layers (features[7] and features[8])
        for i in [7, 8]:
            for param in self.model.features[i].parameters():
                param.requires_grad = True

        # Ensure classifier layers are unfrozen
        for param in self.classifier.parameters():
            param.requires_grad = True


class Vgg16(nn.Module):
    def __init__(self, num_classes, pretrained=True, fine_tune=True):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        
        # Freeze layers if not fine-tuning
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
            
        # Modify the classifier
        num_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Identity()
        self.classifier = nn.Sequential(
                            nn.Linear(num_features, 4096),
                            nn.ReLU(),
                            #nn.BatchNorm1d(4096),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(),
                            #nn.BatchNorm1d(2048),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, num_classes)
                        )

        print(f"Model: {self.__class__.__name__}")
        print(f'[INFO]: Number of in_features: {num_features}')
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        else:
            print('[INFO]: Freezing hidden layers...')
            self.freeze_layers()
        
    def forward(self, x):
        # Extract features from the base model
        x = self.model(x)  # Here, x will be the output from the average pooling layer
        # Flatten the features
        x = torch.flatten(x, 1)
        # Pass features through the new classifier layers
        x = self.classifier(x)
        return x
    
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for i in [26, 28]:
            for param in self.model.features[i].parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
 

            

def create_resnet18(classifier: torch.nn, fine_tune=False, pretrained=True):
    model = Resnet18(classifier=classifier, pretrained=pretrained, fine_tune=fine_tune)
    return model

def create_resnet50(classifier: torch.nn, fine_tune=False, pretrained=True):
    model = Resnet50(classifier=classifier, pretrained=pretrained, fine_tune=fine_tune)
    return model

def create_resnet101(classifier: torch.nn, fine_tune=False, pretrained=True):
    model = Resnet101(classifier=classifier, pretrained=pretrained, fine_tune=fine_tune)
    return model

def create_efficientNetB0(classifier: torch.nn, fine_tune=False, pretrained=True):
    model = EfficientNetB0(classifier=classifier, pretrained=pretrained, fine_tune=fine_tune)
    return model

def create_efficientNetB7(classifier: torch.nn, fine_tune=False, pretrained=True):
    model = EfficientNetB7(classifier=classifier,  pretrained=pretrained, fine_tune=fine_tune)
    return model

def create_vgg16(num_classes: int, fine_tune=False, pretrained=True):
    model = Vgg16(num_classes=num_classes, pretrained=pretrained, fine_tune=fine_tune)
    return model

def create_model(model:str, config: dict, num_classes: int):
    fine_tune = config['Fine_Tune']
    pretrained = config['Pretrained']

    if model== 'Resnet18':
        set_classifier_layer(num_features=512, num_classes=num_classes)
        classifier=get_classifier_layer()
        return create_resnet18(classifier=classifier, fine_tune=fine_tune, pretrained=pretrained)
    elif model== 'Resnet50':
        set_classifier_layer(num_features=2048, num_classes=num_classes)
        classifier=get_classifier_layer()
        return create_resnet50(classifier=classifier, fine_tune=fine_tune, pretrained=pretrained)
    elif model== 'Resnet101':
        set_classifier_layer(num_features=2048, num_classes=num_classes)
        classifier=get_classifier_layer()
        return create_resnet101(classifier=classifier, fine_tune=fine_tune, pretrained=pretrained)
    elif model== 'EfficientNetB0':
        set_classifier_layer(num_features=1280, num_classes=num_classes)
        classifier=get_classifier_layer()
        return create_efficientNetB0(classifier=classifier, num_classes=num_classes, fine_tune=fine_tune, pretrained=pretrained)
    elif model== 'EfficientNetB7':
        set_classifier_layer(num_features=2560, num_classes=num_classes)
        classifier=get_classifier_layer()
        return create_efficientNetB7(classifier=classifier,fine_tune=fine_tune, pretrained=pretrained)
    elif model== 'Vgg16':
        return create_vgg16(num_classes=num_classes, fine_tune=fine_tune, pretrained=pretrained)
    else:
        raise ValueError("Model not recognized. Please choose 'Resnet18', 'Resnet50', 'Resnet101', 'EfficientNetB0', 'EfficientNetB7' or 'Vgg16'.")
        
    