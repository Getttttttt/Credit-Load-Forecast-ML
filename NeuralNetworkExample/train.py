import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from myDateset import MyDataset 
from myNet import MyNet  

# 构造读取器
train_dataset = MyDataset('./NeuralNetworkExample/data/train_public.csv')

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1000,
    shuffle=True,
    drop_last=False)

# 构造模型
model = MyNet()

# 如果有保存的模型，加载模型
# model.load_state_dict(torch.load('model.pth'))

model.train()

max_epoch = 30
opt = optim.Adam(model.parameters(), lr=0.001)

# 训练
now_step = 0
for epoch in range(max_epoch):
    for step, (data, emb_data, label) in enumerate(train_dataloader):
        now_step += 1

        # 转换label为浮点型以匹配BCEWithLogitsLoss的要求
        label = label.float().view(-1, 1)

        # 前向传播
        pre = model(data, emb_data)

        # 计算损失
        # 使用BCEWithLogitsLoss，因为这是一个二分类问题
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))  # 假设正样本的权重是5
        loss = criterion(pre, label)

        # 反向传播和优化
        opt.zero_grad()
        loss.backward()
        opt.step()

        if now_step % 1 == 0:
            print("epoch: {}, batch: {}, loss is: {}".format(epoch, step, loss.item()))

# 保存模型到model.pth
torch.save(model.state_dict(), './NeuralNetworkExample/model/model04301620.pth')
