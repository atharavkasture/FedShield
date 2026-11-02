import copy
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

def train_client_resnet_dp(
    local_model,
    global_model,
    train_loader,
    mu,
    device,
    epochs=2,
    lr=0.1,
    noise_multiplier=1.2,
    max_grad_norm=1.0
):
    """
    Trains local linear classification head using Opacus DP-SGD and proximal regularization.
    """
    local_model.train()
    local_model.to(device)
    global_model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Optimize only the trainable parameters (fc layer)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, local_model.parameters()),
        lr=lr,
        momentum=0.9
    )

    # Attach Opacus Privacy Engine
    privacy_engine = PrivacyEngine()
    model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm
    )

    for epoch in range(epochs):
        for data, target in loader_dp:
            data = data.to(device)
            target = target.view(-1).long().to(device)

            optimizer_dp.zero_grad()
            output = model_dp(data)
            loss = criterion(output, target)

            # Add Proximal Penalty if mu > 0
            if mu > 0:
                proximal_term = 0.0
                for local_p, global_p in zip(model_dp.parameters(), global_model.parameters()):
                    if local_p.requires_grad:
                        proximal_term += ((local_p - global_p) ** 2).sum()
                loss += (mu / 2.0) * proximal_term

            loss.backward()
            optimizer_dp.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    return model_dp, epsilon


# NO DP - Implementing Normal ML/DL principles
def train_client_resnet_nodp(
    local_model,
    global_model,
    train_loader,
    mu,
    device,
    epochs=2,
    lr=0.1
):
    """
    Standard PyTorch training loop WITHOUT Differential Privacy.
    Includes L2 Regularization and Cosine Annealing Learning Rate.
    """
    local_model.train()
    local_model.to(device)
    global_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Added weight_decay for structural penalty
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, local_model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4 
    )

    # Cosine Annealing drops LR smoothly over the course of local batches
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    for epoch in range(epochs):
        for data, target in train_loader:
            data = data.to(device)
            target = target.view(-1).long().to(device)
            
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)

            # Add Proximal Penalty if mu > 0 (FedProx & FedShield logic)
            if mu > 0:
                proximal_term = 0.0
                for local_p, global_p in zip(local_model.parameters(), global_model.parameters()):
                    if local_p.requires_grad:
                        proximal_term += ((local_p - global_p) ** 2).sum()
                loss += (mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()
            scheduler.step()

    return local_model, 0.0