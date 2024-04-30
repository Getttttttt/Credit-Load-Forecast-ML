import torch
import torch.nn.functional as F
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Linear(in_features=21, out_features=512)
        
        self.bn1 = nn.BatchNorm1d(512)  # 批量归一化层
        self.emb1 = nn.Linear(in_features=10, out_features=512)
        self.bn2 = nn.BatchNorm1d(512)  # 批量归一化层
        self.emb2 = nn.Linear(in_features=512, out_features=1024)
        self.bn3 = nn.BatchNorm1d(1024)  # 批量归一化层
        self.emb3 = nn.Linear(in_features=1024, out_features=512)
        self.bn4 = nn.BatchNorm1d(512)  # 批量归一化层
        self.emb4 = nn.Linear(in_features=512, out_features=256)
        
        self.dropout = nn.Dropout(0.5)  # Dropout层
        self.out = nn.Linear(in_features=768, out_features=1)  # 注意输出层输入特征数的调整

    def forward(self, data, emb_data):
        x = self.fc(data)
        x = F.relu(self.bn1(x))  # 添加ReLU激活函数和批量归一化
        
        emb = self.emb1(emb_data)
        emb = F.relu(self.bn2(emb))
        emb = self.emb2(emb)
        emb = F.relu(self.bn3(emb))
        emb = self.emb3(emb)
        emb = F.relu(self.bn4(emb))
        emb = self.emb4(emb)

        x = torch.cat((x, emb), dim=-1)  # 确保维度正确
        
        x = self.dropout(x)  # 添加dropout
        x = self.out(x)
        
        return torch.sigmoid(x)