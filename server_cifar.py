import torch

def federated_averaging(global_model, client_models, weights=None):
    """
    Averages the weights of the models from all clients.
    If 'weights' are provided (e.g., Shannon Entropy), performs an Entropy-Weighted Average.
    Safely strips the '_module.' prefix added by Opacus.
    """
    global_dict = global_model.state_dict()
    
    # If no weights are provided (FedAvg, FedProx), use equal uniform weights
    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)
    else:
        # Normalize provided weights (like Entropy) so they sum to 1.0
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
    for k in global_dict.keys():
        client_tensors = []
        for i in range(len(client_models)):
            client_dict = client_models[i].state_dict()
            
            # Opacus maps keys to '_module.<layer_name>'
            client_key = f"_module.{k}" if f"_module.{k}" in client_dict else k
            
            # Multiply the client's tensor by its normalized weight
            weighted_tensor = client_dict[client_key].float() * weights[i]
            client_tensors.append(weighted_tensor)
            
        # Sum the weighted tensors (instead of a blind mean)
        global_dict[k] = torch.stack(client_tensors, 0).sum(0)
        
    global_model.load_state_dict(global_dict)
    return global_model