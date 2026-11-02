import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Changing batch_size default from 32 to 64 (or 128) right here for faster training. Since CIFAR-10 has 50000.
def get_federated_dataloaders(num_clients=5, alpha=0.1, batch_size=64):
    """
    Loads CIFAR-10, resizes to 224x224 for ResNet-18, and creates Non-IID Dirichlet splits.
    Includes data augmentation for the training set.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Downloading/Loading CIFAR-10 dataset with Augmentation...")
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)

    # Dirichlet Non-IID Partitioning
    labels = np.array(train_dataset.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < len(labels) / num_clients) for p, idx_j in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        splits = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    # 2. Add num_workers=2 and pin_memory=True to both DataLoaders right here:
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        client_loaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    print("Data Partitioning Complete across 5 Organizations.")
    return client_loaders, test_loader