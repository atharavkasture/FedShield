# # Step 1 in coding - tells total transactions and fraud cases in the dataset
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.model_selection import train_test_split

# # def load_and_preprocess_data(filepath="data/creditcard.csv"):
# #     print("Loading dataset...")
# #     # Load the CSV
# #     df = pd.read_csv(filepath)
    
# #     # The 'Amount' and 'Time' columns need to be scaled to match the V1-V28 PCA columns
# #     scaler = StandardScaler()
# #     df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
# #     df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
# #     # Separate features (X) and labels (y)
# #     X = df.drop('Class', axis=1).values
# #     y = df['Class'].values
    
# #     # Split into a global Train and Test set (80% train, 20% test)
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
# #     print(f"Data Loaded Successfully!")
# #     print(f"Total Transactions: {len(df)}")
# #     print(f"Total Fraud Cases: {np.sum(y)}")
# #     print(f"Training set shape: {X_train.shape}")
# #     print(f"Testing set shape: {X_test.shape}")
    
# #     return X_train, X_test, y_train, y_test

# # if __name__ == "__main__":
# #     # Test the function to make sure it works
# #     X_train, X_test, y_train, y_test = load_and_preprocess_data()

# # Step 3 in coding - creates a PyTorch Dataset and DataLoader for each client, simulating Non-IID data distribution using Dirichlet distribution
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# class FinancialDataset(Dataset):
#     """PyTorch Dataset wrapper for financial tabular records."""
#     def __init__(self, features, labels):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.long)
        
#     def __len__(self):
#         return len(self.features)
        
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# def partition_non_iid_dirichlet(y_train, num_clients=5, alpha=0.5, seed=42):
#     """
#     Partitions the dataset across clients using a Dirichlet distribution 
#     to simulate severe real-world label imbalance (Non-IID).
#     """
#     np.random.seed(seed)
#     num_classes = len(np.unique(y_train))
#     client_indices = [[] for _ in range(num_clients)]
    
#     for c in range(num_classes):
#         # Get all indices belonging to class c (0 or 1)
#         idx_c = np.where(y_train == c)[0]
#         np.random.shuffle(idx_c)
        
#         # Sample proportions from Dirichlet distribution
#         proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
#         # Normalize and scale to match number of samples in class c
#         proportions = np.cumsum((proportions * len(idx_c)).astype(int)[:-1])
#         splits = np.split(idx_c, proportions)
        
#         for client_id in range(num_clients):
#             client_indices[client_id].extend(splits[client_id])
            
#     return client_indices

# def get_federated_dataloaders(filepath="data/creditcard.csv", num_clients=5, alpha=0.5, batch_size=64):
#     print("Loading and preparing federated partitions...")
#     df = pd.read_csv(filepath)
    
#     # Scale Amount and Time features
#     scaler = StandardScaler()
#     df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
#     df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
#     X = df.drop('Class', axis=1).values
#     y = df['Class'].values
    
#     # 80-20 Train-Test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     # Generate Non-IID partitions
#     client_indices = partition_non_iid_dirichlet(y_train, num_clients=num_clients, alpha=alpha)
    
#     client_loaders = []
#     print("\n--- Virtual Bank Partitions (Non-IID Profile) ---")
#     for client_id, indices in enumerate(client_indices):
#         client_X = X_train[indices]
#         client_y = y_train[indices]
        
#         num_fraud = np.sum(client_y == 1)
#         num_legit = np.sum(client_y == 0)
#         fraud_ratio = (num_fraud / len(client_y)) * 100 if len(client_y) > 0 else 0
        
#         print(f"Bank {client_id + 1}: Total = {len(client_y):6d} | Legit = {num_legit:6d} | Fraud = {num_fraud:3d} ({fraud_ratio:.3f}%)")
        
#         dataset = FinancialDataset(client_X, client_y)
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         client_loaders.append(loader)
        
#     # Global test loader for the server to evaluate performance
#     test_dataset = FinancialDataset(X_test, y_test)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     return client_loaders, test_loader

# if __name__ == "__main__":
#     client_loaders, test_loader = get_federated_dataloaders()

# After the Skin dataset, the complete FedShield Architecture is now implemented.
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FinancialDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def partition_non_iid_dirichlet(y_train, num_clients=5, alpha=0.5, seed=42):
    np.random.seed(seed)
    num_classes = len(np.unique(y_train))
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_c = np.where(y_train == c)[0]
        np.random.shuffle(idx_c)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.cumsum((proportions * len(idx_c)).astype(int)[:-1])
        splits = np.split(idx_c, proportions)
        
        for client_id in range(num_clients):
            client_indices[client_id].extend(splits[client_id])
            
    return client_indices

def get_federated_dataloaders(filepath="data/creditcard.csv", num_clients=5, alpha=0.1, batch_size=64):
    print("Loading and preparing federated partitions...")
    df = pd.read_csv(filepath)
    
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    client_indices = partition_non_iid_dirichlet(y_train, num_clients=num_clients, alpha=alpha)
    client_loaders = []
    
    for client_id, indices in enumerate(client_indices):
        dataset = FinancialDataset(X_train[indices], y_train[indices])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
        
    test_dataset = FinancialDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader