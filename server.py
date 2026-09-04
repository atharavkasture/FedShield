# # Step 6 in coding - implements the server-side logic for federated learning, including model aggregation (FedAvg), evaluation, and the main training loop that orchestrates the entire federated learning process
# import torch
# import copy
# import numpy as np
# from sklearn.metrics import average_precision_score, roc_auc_score

# # Import all our custom modules
# from dataset import get_federated_dataloaders
# from models import FraudMLP
# from utils import calculate_entropy, compute_dynamic_mu
# from client import train_client

# def evaluate_model(model, test_loader, device="cpu"):
#     """Tests the global model on unseen data."""
#     model.eval()
#     all_labels = []
#     all_preds = []
    
#     with torch.no_grad():
#         for features, labels in test_loader:
#             features, labels = features.to(device), labels.to(device)
#             outputs = model(features)
#             # Apply softmax to get probabilities for the "Fraud" class (index 1)
#             probs = torch.softmax(outputs, dim=1)[:, 1]
            
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(probs.cpu().numpy())
            
#     # Use PR-AUC instead of accuracy because data is highly imbalanced
#     pr_auc = average_precision_score(all_labels, all_preds)
#     roc_auc = roc_auc_score(all_labels, all_preds)
#     return pr_auc, roc_auc
# # Here the Opacus changes the weight name from what pytorch uses. So it gave error.
# # def federated_averaging(global_model, client_models):
# #     """Averages the weights of all client models to create the new global model."""
# #     global_dict = global_model.state_dict()
    
# #     for k in global_dict.keys():
# #         # Sum the weights of all clients for layer 'k'
# #         global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        
# #     global_model.load_state_dict(global_dict)
# #     return global_model

# def federated_averaging(global_model, client_models):
#     """Averages the weights of all client models to create the new global model."""
#     global_dict = global_model.state_dict()
    
#     for k in global_dict.keys():
#         client_tensors = []
#         for client_model in client_models:
#             # Opacus wraps the model in a GradSampleModule. 
#             # We access the original model using ._module to bypass the wrapper.
#             if hasattr(client_model, '_module'):
#                 client_dict = client_model._module.state_dict()
#             else:
#                 client_dict = client_model.state_dict()
                
#             client_tensors.append(client_dict[k].float())
            
#         # Sum the weights of all clients for layer 'k' and average them
#         global_dict[k] = torch.stack(client_tensors, 0).mean(0)
        
#     global_model.load_state_dict(global_dict)
#     return global_model

# if __name__ == "__main__":
#     print("--- Starting FedShield Server ---")
    
#     # 1. Setup Data
#     num_clients = 5
#     client_loaders, test_loader = get_federated_dataloaders(num_clients=num_clients)
    
#     # 2. Calculate Entropies and Dynamic mu for all clients
#     entropies = []
#     for loader in client_loaders:
#         # Extract all labels from the loader to compute entropy
#         labels = torch.cat([y for _, y in loader]).numpy()
#         entropies.append(calculate_entropy(labels))
        
#     dynamic_mus = compute_dynamic_mu(entropies)
    
#     # 3. Initialize Global Model
#     global_model = FraudMLP()
    
#     # 4. Federated Training Loop
#     COMMUNICATION_ROUNDS = 25  # Set to 5 for a quick test (can be 20+ later)
    
#     for round_num in range(COMMUNICATION_ROUNDS):
#         print(f"\n[Round {round_num + 1}/{COMMUNICATION_ROUNDS}]")
#         client_models = []
        
#         # Train each client
#         for i in range(num_clients):
#             print(f"  -> Training Bank {i+1} (mu_k = {dynamic_mus[i]:.4f})...", end=" ")
            
#             # Make a copy of the global model for the client to train
#             local_model = copy.deepcopy(global_model)
            
#             trained_model, loss, eps = train_client(
#                 model=local_model,
#                 global_model=global_model,
#                 dataloader=client_loaders[i],
#                 mu_k=dynamic_mus[i],
#                 epochs=1
#             )
            
#             client_models.append(trained_model)
#             print(f"Loss: {loss:.4f} | Eps: {eps:.2f}")
            
#         # Aggregate all client models via FedAvg
#         global_model = federated_averaging(global_model, client_models)
        
#         # Evaluate the new global model
#         pr_auc, roc_auc = evaluate_model(global_model, test_loader)
#         print(f"  => Global Model Evaluation: PR-AUC = {pr_auc:.4f} | ROC-AUC = {roc_auc:.4f}")

#     print("\n--- Federated Training Complete! ---")

# After the Skin dataset, the complete FedShield Architecture is now implemented.

# For Accuracy and precsion
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score
'''
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    model.to(device)
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            # Probabilities for AUC
            probs = torch.softmax(outputs, dim=1)[:, 1]
            # Hard predictions (0 or 1) for Accuracy/Precision
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    pr_auc = average_precision_score(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # Calculate metrics specifically for Fraud (zero_division=0 prevents warnings if no fraud is predicted)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    return pr_auc, roc_auc, accuracy, precision, recall
'''

def evaluate_model(model, test_loader, device="cpu", return_preds=False):
    model.eval()
    model.to(device)
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    pr_auc = average_precision_score(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    # We added this block so the notebook can grab the labels for the diagram
    if return_preds:
        return pr_auc, roc_auc, accuracy, precision, recall, all_labels, all_preds
        
    return pr_auc, roc_auc, accuracy, precision, recall

def federated_averaging(global_model, client_models, weights=None):
    global_dict = global_model.state_dict()
    
    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)
    else:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
    for k in global_dict.keys():
        client_tensors = []
        for i, client_model in enumerate(client_models):
            client_dict = client_model._module.state_dict() if hasattr(client_model, '_module') else client_model.state_dict()
            weighted_tensor = client_dict[k].float() * weights[i]
            client_tensors.append(weighted_tensor)
            
        global_dict[k] = torch.stack(client_tensors, 0).sum(0)
        
    global_model.load_state_dict(global_dict)
    return global_model