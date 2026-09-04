import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F

def calculate_entropy(labels, num_classes=7):
    """Calculates Shannon Entropy (H_k) for a clinic's label distribution."""
    counts = np.bincount(labels, minlength=num_classes)
    probabilities = counts / len(labels)
    # Filter out zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def compute_dynamic_mu(entropies, num_classes=7, mu_base=0.01, lambda_scale=1.5):
    """
    Dynamically scales the FedShield regularization penalty:
    mu_k = mu_base * (1 + lambda * (H_max - H_k))
    """
    h_max = np.log2(num_classes)
    dynamic_mus = []
    for h_k in entropies:
        mu_k = mu_base * (1.0 + lambda_scale * (h_max - h_k))
        dynamic_mus.append(mu_k)
    return dynamic_mus
'''
def evaluate_resnet(model, test_loader, device):
    """Evaluates the global model on test data, returning accuracy and multi-class One-vs-One ROC-AUC."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            
            y = y.view(-1).to(device)
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    
    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class="ovo", average="macro")
    except Exception:
        auc = 0.50

    return acc, auc
'''

def evaluate_resnet(model, test_loader, device, return_preds=False):
    """Evaluates the global model on test data, returning accuracy and multi-class One-vs-One ROC-AUC."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            
            y = y.view(-1).to(device)
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    
    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class="ovo", average="macro")
    except Exception:
        auc = 0.50

    # NEW: Return raw arrays if requested for the final round visuals
    if return_preds:
        return acc, auc, all_targets, all_preds

    return acc, auc