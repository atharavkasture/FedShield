import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import medmnist
from medmnist import DermaMNIST

def get_federated_dataloaders(num_clients=5, alpha=0.5, batch_size=32):
    """
    Loads DermaMNIST, resizes to 224x224 for ResNet-18, and creates Non-IID Dirichlet splits.
    """
    # Transform: Resize 28x28 thumbnails to 224x224 for pre-trained ResNet
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Downloading/Loading DermaMNIST dataset...")
    train_dataset = DermaMNIST(split="train", transform=train_transform, download=True)
    test_dataset = DermaMNIST(split="test", transform=test_transform, download=True)

    # Dirichlet Non-IID Partitioning
    labels = np.array([label[0] for label in train_dataset.labels])
    num_classes = 7
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample class proportions per client from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < len(labels) / num_clients) for p, idx_j in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        splits = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data Partitioning Complete across 5 Virtual Clinics.")
    return client_loaders, test_loader