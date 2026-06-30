"""data.py

Dataset construction, data augmentation pipeline, train / validation
splitting, and DataLoader creation.
"""

import os

import numpy as np
import torch
from torch import Tensor
import torchvision
from torch.utils.data import Subset, Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, Dict


class TransformSubset(Dataset):
    """
    Dataset wrapper that applies a different transform to a subset.
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # For ImageFolder dataset and Subset of ImageFolder
        # if isinstance(self.dataset, Subset):
        #     original_idx = self.dataset.indices[idx]
        #     img_path = self.dataset.dataset.samples[original_idx][0]
        #     img = self.dataset.dataset.loader(img_path)
        # else:
        #     img_path = self.dataset.samples[idx][0]
        #     img = self.dataset.loader(img_path)
        
        return self.transform(img), label
    
    def __len__(self):
        return len(self.dataset)


# function from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        

def get_transforms(args):
    """Build training and evaluation image transformations.

    Training uses a composite augmentation pipeline with geometric
    (TrivialAugmentWideNoColor) and photometric (TrivialAugmentWideNoShape)
    perturbations.  If ``args.data_aug`` is False the training transform
    falls back to the evaluation transform (resize + normalize).

    Args:
        args: Arguments containing:
            - img_size: Image size for resizing.
            - data_aug: Whether to use data augmentation for training.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms_dict = {}

    # Evaluation transformation (same for validation and test)
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transforms_dict['eval'] = eval_transform

    # Training transformation: geometric + photometric augmentation
    train_transform = transforms.Compose([
        transforms.Resize(size=(args.img_size + 8, args.img_size + 8)),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(args.img_size + 4, scale=(0.95, 1.)),
        TrivialAugmentWideNoShape(),
        transforms.RandomCrop(size=(args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transforms_dict['train'] = train_transform

    if not args.data_aug:
        transforms_dict['train'] = eval_transform

    return transforms_dict
        
def get_datasets(args):
    """
    Get train, validation, and test datasets based on provided arguments.
    
    Args:
        args: Arguments containing:
            - data_dir: Root directory containing 'train' and 'test' folders
            - img_size: Image size for resizing (assumed square)
            - num_classes: Number of classes to use
            - data_aug: Whether to use data augmentation for training
    
    Returns:
        train_dataset, val_dataset, test_dataset: The dataset splits
    """
    # Define transformations
    transforms_dict = get_transforms(args)

    # Load initial datasets with respective transforms
    train_data_path = os.path.join(args.data_dir, 'train')
    test_data_path = os.path.join(args.data_dir, 'test')
    
    # We'll apply train_transform later to ensure validation gets eval_transform
    full_train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=None)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transforms_dict['eval'])
    
    # Filter by number of classes
    if args.num_classes < len(full_train_dataset.classes):
        train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label < args.num_classes]
        filtered_train_dataset = Subset(full_train_dataset, train_indices)
        
        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label < args.num_classes]
        test_dataset = Subset(test_dataset, test_indices)
    else:
        filtered_train_dataset = full_train_dataset
    
    # Split into train and validation BEFORE applying transforms
    train_indices, val_indices = split_train_val_indices(filtered_train_dataset, val_ratio=0.2)
    
    # Create train and validation subsets
    train_subset = Subset(filtered_train_dataset, train_indices)
    val_subset = Subset(filtered_train_dataset, val_indices)
    
    # Apply the appropriate transforms to each dataset
    train_dataset = TransformSubset(train_subset, transforms_dict['train'])
    val_dataset = TransformSubset(val_subset, transforms_dict['eval'])
    
    return train_dataset, val_dataset, test_dataset

def split_train_val_indices(dataset, val_ratio=0.2):
    """
    Get indices for stratified split of the dataset into train and validation sets.
    
    Args:
        dataset: Dataset to split
        val_ratio: Ratio for validation set size
    
    Returns:
        train_indices, val_indices: Indices for the train and validation splits
    """
    targets = []
    for i in range(len(dataset)):
        if isinstance(dataset, Subset):
            _, label = dataset.dataset[dataset.indices[i]]
        else:
            _, label = dataset[i]
        targets.append(label)
    
    targets = np.array(targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    
    return train_idx, val_idx


def get_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    args
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing datasets.

    Args:
        train_dataset: Training dataset (shuffle=True).
        val_dataset: Validation dataset (shuffle=False).
        test_dataset: Test dataset (shuffle=False).
        args: Arguments containing:
            batch_size
            num_workers
            pin_memory

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            - train_loader: DataLoader with shuffle enabled.
            - val_loader: DataLoader with shuffle disabled.
            - test_loader: DataLoader with shuffle disabled.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    return train_loader, val_loader, test_loader
