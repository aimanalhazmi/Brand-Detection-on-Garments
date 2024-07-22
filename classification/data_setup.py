
"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os
import torchvision.transforms as T
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import random_split
import random 
import torch
from torch.utils.data import SubsetRandomSampler
#from torch.utils.data import default_collate
#from torchvision.transforms import v2

NUM_WORKERS = os.cpu_count()

def is_valid_file(path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    return os.path.splitext(path)[1].lower() in valid_extensions


def get_datasets(
    data_dir: str,
    split_size: float = 0.8):
    dataset = datasets.ImageFolder(data_dir, is_valid_file=is_valid_file,  transform=None)
    print("[INFO]: Dataset loaded successfully.")
    class_names = dataset.classes
    random.seed(42)
        # Split the dataset into train and test sets
    train_size = int(split_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, valid_dataset, class_names
    
def get_dataset(data_dir: str):

    dataset = datasets.ImageFolder(data_dir, is_valid_file=is_valid_file,  transform=None)
    print("[INFO]: Dataset loaded successfully.")
    class_names = dataset.classes
    return dataset, class_names
    


def print_dataset_info(dataset, kf):
    for fold, (train_ids, valid_ids) in enumerate(kf.split(np.arange(len(dataset)))):
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)
        transforms_train, transforms_valid = get_transforms()

        # Create data loaders for training and validation
        train_dataset = CustomDataset(dataset, transform=transforms_train)
        valid_dataset = CustomDataset(dataset, transform=transforms_valid)
        print('[INFO]: Dataset size:', len(train_dataset))
        print('[INFO]: Train dataset size:', len(train_subsampler))
        print('[INFO]: Valid dataset size:', len(valid_subsampler))
        class_names = dataset.classes
        num_classes = len(class_names)
        print('[INFO]: Number of Classes:', num_classes)
        break
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    
class PadTensor(object):
    def __init__(self, padding):
        # Padding can be a single integer or a tuple (left, right, top, bottom)
        self.padding = padding  

    def __call__(self, tensor):
        # Pad the tensor and specify the padding mode
        # 'constant' mode adds constant valued padding, and 0 is the padding value
        return torch.nn.functional.pad(tensor, self.padding, mode='constant', value=0)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_transforms():
    set_seed(42)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),   
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(p=0.5), # data augmentation
    #transforms.RandomRotation(45),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    #transforms.Lambda(lambda image: image.convert('RGB')),
    #transforms.Grayscale(num_output_channels=3),  
    #transforms.GaussianBlur(kernel_size=(3, 5), sigma=(1, 2)),
    transforms.ToTensor(),
    #PadTensor(padding=(1, 1, 1, 1)),  # Applies padding to the tensor
    transforms.Normalize(mean, std) # normalization
    ])
    transforms_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.CenterCrop((224, 224)),
    #transforms.RandomRotation(45),
    transforms.ToTensor(),
    #PadTensor(padding=(1, 1, 1, 1)),  # Applies padding to the tensor
    transforms.Normalize(mean, std)
    ])
    print("[INFO]: Transforms are Done ..")
    return transforms_train, transforms_valid


def collate_fn(batch):
    cutmix = v2.CutMix(num_classes=286)
    mixup = v2.MixUp(num_classes=286)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    return cutmix_or_mixup(*default_collate(batch))


def std_transform():
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])
    
def create_dataloaders(
    train_dataset, 
    valid_dataset,
    batch_size: int=32, 
    augmented: bool=True,
    num_workers: int=NUM_WORKERS):
    
    transforms_train, transforms_valid = get_transforms()
    if augmented:
            # Create datasets with appropriate transformations
        train_dataset = CustomDataset(train_dataset, transform=transforms_train)
        valid_dataset = CustomDataset(valid_dataset, transform=transforms_valid)
    else:
        train_dataset = CustomDataset(train_dataset, transform=std_transform())
        valid_dataset = CustomDataset(valid_dataset, transform=std_transform())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) #num_workers=0 for reproducibility
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataloader, valid_dataloader


def get_dataloaders(train_dataset, valid_dataset, batch_size: int=32, augmented: bool=True, num_workers: int=NUM_WORKERS):
    train_dataloader, valid_dataloader = create_dataloaders(train_dataset, valid_dataset, batch_size, augmented, num_workers)
    return  train_dataloader, valid_dataloader

def get_dataloaders_for_cross_validation(dataset, train_ids, valid_ids, batch_size:int=32, num_workers: int=NUM_WORKERS):
    train_subsampler = SubsetRandomSampler(train_ids)
    valid_subsampler = SubsetRandomSampler(valid_ids)
    transforms_train, transforms_valid = get_transforms()
        # Create data loaders for training and validation
    train_dataset = CustomDataset(dataset, transform=transforms_train)
    valid_dataset = CustomDataset(dataset, transform=transforms_valid)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_subsampler, num_workers=num_workers)
    return train_dataloader, valid_dataloader