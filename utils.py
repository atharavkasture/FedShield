# # Step 4 in coding - calculates the entropy of each client's data distribution and assigns a dynamic mu_k for proximal regularization based on the entropy
# import numpy as np

# def calculate_entropy(y_labels):
#     """
#     Calculates Shannon Entropy of the class distribution.
#     Formula: H = -sum(p * log2(p))
#     """
#     if len(y_labels) == 0:
#         return 0.0
    
#     # Count occurrences of each class (0 and 1)
#     _, counts = np.unique(y_labels, return_counts=True)
#     probabilities = counts / len(y_labels)
    
#     # Add 1e-9 to prevent log2(0) mathematical errors (like in Bank 4)
#     entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
#     return entropy

# def compute_dynamic_mu(client_entropies, base_mu=0.01, lambda_penalty=1.0):
#     """
#     Calculates dynamic mu_k for each client based on their entropy.
#     Skewed clients get a higher mu (tighter rubber band).
#     """
#     max_entropy = np.max(client_entropies)
    
#     dynamic_mus = []
#     for h in client_entropies:
#         # Formula: mu_k = base_mu * (1 + lambda * (H_max - H_k))
#         mu_k = base_mu * (1 + lambda_penalty * (max_entropy - h))
#         dynamic_mus.append(mu_k)
        
#     return dynamic_mus

# if __name__ == "__main__":
#     # Testing with the exact numbers from your terminal output!
#     bank_distributions = [
#         np.array([0]*34713 + [1]*1),     # Bank 1
#         np.array([0]*10251 + [1]*388),   # Bank 2 (Most balanced)
#         np.array([0]*10101 + [1]*3),     # Bank 3
#         np.array([0]*153016 + [1]*0),    # Bank 4 (Most skewed)
#         np.array([0]*19370 + [1]*2)      # Bank 5
#     ]
    
#     entropies = [calculate_entropy(y) for y in bank_distributions]
#     mus = compute_dynamic_mu(entropies)
    
#     print("--- Dynamic Proximal Regularization (mu_k) ---")
#     for i, (h, mu) in enumerate(zip(entropies, mus)):
#         print(f"Bank {i+1}: Entropy = {h:.4f} | Assigned mu_k = {mu:.5f}")


# After the Skin dataset, the complete FedShield Architecture is now implemented.
import numpy as np

def calculate_entropy(y_labels, num_classes=2):
    """Calculates Shannon Entropy for binary classification (Fraud vs Normal)."""
    if len(y_labels) == 0:
        return 0.0
    _, counts = np.unique(y_labels, return_counts=True)
    probabilities = counts / len(y_labels)
    # Filter zeros safely to prevent log2 errors
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def compute_dynamic_mu(entropies, num_classes=2, mu_base=0.01, lambda_scale=0.5):
    """Scales FedShield penalty dynamically: mu_k = mu_base * (1 + lambda * (H_max - H_k))"""
    h_max = np.log2(num_classes) # For 2 classes, max entropy is exactly 1.0
    dynamic_mus = []
    for h_k in entropies:
        mu_k = mu_base * (1.0 + lambda_scale * (h_max - h_k))
        dynamic_mus.append(mu_k)
    return dynamic_mus