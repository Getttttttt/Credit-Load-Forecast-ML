import torch
from torch.utils.data import DataLoader
import pandas as pd
from myDateset import MyDataset  
from myNet import MyNet  

# 初始化模型
model = MyNet()

# 加载模型参数
model.load_state_dict(torch.load('./NeuralNetworkExample/model/model04301620.pth'))

model.eval()

# 初始化测试数据集
test_dataset = MyDataset('./NeuralNetworkExample/data/test_public.csv', mode='test')

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False)

# 预测并保存结果
result = []
for step, (data, emb_data, loan_id) in enumerate(test_dataloader):
    with torch.no_grad():  # 确保不计算梯度
        pre = model(data, emb_data)
        # 使用 Sigmoid 函数将输出转换为概率
        prob = torch.sigmoid(pre).item()  # 获取第二列（正类）的概率值
        result.append([loan_id.item(), prob])

# 保存结果到 CSV 文件
pd.DataFrame(result, columns=['id', 'isDefault']).to_csv('../NeuralNetworkExample/submission.csv', index=None)
