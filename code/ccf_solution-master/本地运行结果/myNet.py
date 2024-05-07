import torch
import torch.nn.functional as F
import torch.nn as nn

'''class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Linear(in_features=21, out_features=512)
        self.bn_fc = nn.BatchNorm1d(512)  # Batch normalization for the fully connected layer
        self.dropout_fc = nn.Dropout(0.5)  # Dropout to prevent overfitting

        self.emb1 = nn.Linear(in_features=10, out_features=2048)
        self.bn_emb1 = nn.BatchNorm1d(2048)  # Batch normalization for the first embedding layer
        self.emb2 = nn.Linear(in_features=2048, out_features=512)
        self.bn_emb2 = nn.BatchNorm1d(512)  # Batch normalization for the second embedding layer
        self.dropout_emb = nn.Dropout(0.5)  # Dropout for embeddings

        self.out = nn.Linear(in_features=1024, out_features=1)

    def forward(self, data, emb_data):
        x = F.relu(self.fc(data))  # Apply ReLU activation
        x = self.bn_fc(x)  # Apply batch normalization
        x = self.dropout_fc(x)  # Apply dropout

        emb = F.relu(self.emb1(emb_data))  # Apply ReLU activation
        emb = self.bn_emb1(emb)  # Apply batch normalization
        emb = F.relu(self.emb2(emb))  # Apply ReLU activation
        emb = self.bn_emb2(emb)  # Apply batch normalization
        emb = self.dropout_emb(emb)  # Apply dropout

        x = torch.cat((x, emb), dim=-1)  # Concatenate feature representations

        x = self.out(x)  # Final output layer
        
        return x'''


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(in_features=21, out_features=512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)  # Additional layer
        self.bn_fc2 = nn.BatchNorm1d(512)
        
        self.dropout_fc = nn.Dropout(0.5)
        
        self.emb1 = nn.Linear(in_features=10, out_features=2048)
        self.bn_emb1 = nn.BatchNorm1d(2048)
        self.emb2 = nn.Linear(in_features=2048, out_features=512)
        self.bn_emb2 = nn.BatchNorm1d(512)
        self.dropout_emb = nn.Dropout(0.5)
        
        self.out = nn.Linear(in_features=1024, out_features=1)
        # 初始化权重的正则化
        weight_decay=1e-4
        self.weight_decay = weight_decay
        

    def forward(self, data, emb_data):
        x = F.leaky_relu(self.fc1(data))
        x = self.bn_fc1(x)
        x = F.leaky_relu(self.fc2(x))  # Additional layer with activation
        x = self.bn_fc2(x)
        x = self.dropout_fc(x)
        
        emb = F.leaky_relu(self.emb1(emb_data))
        emb = self.bn_emb1(emb)
        emb = F.leaky_relu(self.emb2(emb))
        emb = self.bn_emb2(emb)
        emb = self.dropout_emb(emb)
        
        x = torch.cat((x, emb), dim=-1)
        
        x = self.out(x)
        
        return x

