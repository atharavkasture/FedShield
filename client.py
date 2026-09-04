# # Step 5 in coding - implements the local training loop for a single client, incorporating both Dynamic FedProx and Differential Privacy (DP-SGD) using Opacus
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from opacus import PrivacyEngine
# import copy

# # Import our previously written modules
# from models import FraudMLP
# from dataset import get_federated_dataloaders

# def train_client(model, global_model, dataloader, mu_k, epochs=1, device="cpu"):
#     """
#     Trains a single client's model locally using Dynamic FedProx and DP-SGD.
#     """
#     model.train()
#     global_model.eval() # Global model is frozen, just used for the rubber band math
    
#     # criterion = nn.CrossEntropyLoss() changed this to:
#     # We heavily penalize the model for missing a fraud case (Class 1)
#     # The ratio is roughly 1 legitimate : 100 fraud penalty
#     class_weights = torch.tensor([1.0, 100.0]).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)



#     # We only train the classification layers for "Layer-Wise" DP
#     optimizer = optim.SGD(model.parameters(), lr=0.005) # Here changed the learning rate to 0.005 from 0.01 to reduce the noise impact on DP-SGD
    
#     # ---------------------------------------------------------
#     # NOVELTY 2: DIFFERENTIAL PRIVACY (DP-SGD)
#     # ---------------------------------------------------------
#     # We attach the Opacus Privacy Engine to inject noise and clip gradients
#     privacy_engine = PrivacyEngine()
#     model, optimizer, dataloader = privacy_engine.make_private(
#         module=model,
#         optimizer=optimizer,
#         data_loader=dataloader,
#         noise_multiplier=1.1, # Controls the epsilon (privacy budget)
#         max_grad_norm=1.0,    # Clips gradients so no single transaction dominates
#     )
    
#     epoch_loss = 0.0
#     for epoch in range(epochs):
#         for batch_features, batch_labels in dataloader:
#             batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(batch_features)
            
#             # Standard classification loss
#             loss = criterion(outputs, batch_labels)
            
#             # ---------------------------------------------------------
#             # NOVELTY 1: DYNAMIC PROXIMAL REGULARIZATION (FedProx)
#             # ---------------------------------------------------------
#             proximal_term = 0.0
#             for local_param, global_param in zip(model.parameters(), global_model.parameters()):
#                 # Calculate the squared distance between local and global weights
#                 proximal_term += ((local_param - global_param) ** 2).sum()
                
#             # Add the "rubber band" penalty to the total loss
#             loss += (mu_k / 2) * proximal_term
            
#             # Backpropagation (Opacus automatically adds the DP noise here)
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
            
#     # Calculate how much privacy budget (epsilon) we used
#     epsilon = privacy_engine.get_epsilon(delta=1e-5)
    
#     return model, epoch_loss / len(dataloader), epsilon

# if __name__ == "__main__":
#     print("Testing Local Client Training...")
    
#     # 1. Get the data
#     client_loaders, _ = get_federated_dataloaders(num_clients=5)
#     bank_1_loader = client_loaders[0]
    
#     # 2. Setup the models
#     global_model = FraudMLP()
#     local_model = copy.deepcopy(global_model)
    
#     # 3. Train Bank 1 for a single epoch (testing our dynamic mu of 0.01225 from earlier)
#     print("\nStarting Training for Bank 1 (with DP and FedProx)...")
#     trained_model, loss, eps = train_client(
#         model=local_model,
#         global_model=global_model,
#         dataloader=bank_1_loader,
#         mu_k=0.01225, 
#         epochs=1
#     )
    
#     print(f"Bank 1 Training Complete!")
#     print(f"Final Loss: {loss:.4f}")
#     print(f"Privacy Budget Consumed (Epsilon): {eps:.4f} (Goal is < 1.5)")

# After the Skin dataset, the complete FedShield Architecture is now implemented.
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

def train_client_dp(local_model, global_model, dataloader, mu_k, device, epochs=1, lr=0.005):
    """Trains classification head using Opacus DP-SGD and proximal regularization."""
    local_model.train()
    local_model.to(device)
    global_model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, local_model.parameters()), lr=lr)
    
    privacy_engine = PrivacyEngine()
    model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )
    
    epoch_loss = 0.0
    for epoch in range(epochs):
        for features, labels in loader_dp:
            features, labels = features.to(device), labels.to(device)
            
            optimizer_dp.zero_grad()
            outputs = model_dp(features)
            loss = criterion(outputs, labels)
            
            if mu_k > 0:
                proximal_term = 0.0
                for local_p, global_p in zip(model_dp.parameters(), global_model.parameters()):
                    if local_p.requires_grad:
                        proximal_term += ((local_p - global_p) ** 2).sum()
                loss += (mu_k / 2.0) * proximal_term
                
            loss.backward()
            optimizer_dp.step()
            epoch_loss += loss.item()
            
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    return model_dp, epoch_loss / len(dataloader), epsilon

def train_client_nodp(local_model, global_model, dataloader, mu_k, device, epochs=1, lr=0.005):
    """Standard PyTorch training WITHOUT Differential Privacy (For Ablation Study)."""
    local_model.train()
    local_model.to(device)
    global_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, local_model.parameters()), lr=lr)
    
    epoch_loss = 0.0
    for epoch in range(epochs):
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = local_model(features)
            loss = criterion(outputs, labels)
            
            if mu_k > 0:
                proximal_term = 0.0
                for local_p, global_p in zip(local_model.parameters(), global_model.parameters()):
                    if local_p.requires_grad:
                        proximal_term += ((local_p - global_p) ** 2).sum()
                loss += (mu_k / 2.0) * proximal_term
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    return local_model, epoch_loss / len(dataloader), 0.0