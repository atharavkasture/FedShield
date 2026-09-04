# # Step 2 in coding - a basic MLP to know the shape of the output and to check if the model is working fine
# import torch
# import torch.nn as nn

# class FraudMLP(nn.Module):
#     def __init__(self, input_dim=30, hidden_dim=64):
#         super(FraudMLP, self).__init__()
        
#         # Layer 1: 30 Input features -> 64 Hidden nodes
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.dropout = nn.Dropout(0.2) # Drops 20% of nodes randomly to prevent overfitting
        
#         # Layer 2: 64 Hidden nodes -> 32 Hidden nodes
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.relu2 = nn.ReLU()
        
#         # Layer 3: 32 Hidden nodes -> 2 Output classes (Legitimate vs Fraud)
#         self.fc3 = nn.Linear(hidden_dim // 2, 2)

#     def forward(self, x):
#         # Pass data through Layer 1
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.dropout(x)
        
#         # Pass data through Layer 2
#         x = self.fc2(x)
#         x = self.relu2(x)
        
#         # Pass data through Layer 3 (Output)
#         x = self.fc3(x)
        
#         # Note: We do NOT use Softmax here because PyTorch's CrossEntropyLoss 
#         # applies it automatically during training.
#         return x

# if __name__ == "__main__":
#     # Let's test the model to make sure it compiles
#     model = FraudMLP()
    
#     # Create 5 fake transactions, each with 30 features
#     dummy_input = torch.randn(5, 30) 
    
#     # Pass them through the network
#     output = model(dummy_input)
    
#     print("Model Output Shape:", output.shape) 
#     print("If the shape is torch.Size([5, 2]), your Neural Network is working perfectly!")

# After the Skin dataset, the complete FedShield Architecture is now implemented.
import torch
import torch.nn as nn

class FraudMLP(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, freeze_backbone=True):
        super(FraudMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        
        # Freeze feature extraction to protect against DP noise
        if freeze_backbone:
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.fc2.parameters():
                param.requires_grad = False
                
        # Final classification head (Trainable)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)
        self.fc3.requires_grad_(True)

    def forward(self, x):
        x = self.dropout(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)