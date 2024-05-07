import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, csv_dir, standard_csv_dir='./SVM/data/train_public.csv', mode='test'):
        super(MyDataset, self).__init__()
        self.df = pd.read_csv(csv_dir)
        st_df = pd.read_csv(standard_csv_dir)
        st_df = st_df.apply(pd.to_numeric, errors='coerce')
        self.mean_df = st_df.mean()
        self.std_df = st_df.std()

        self.num_item = ['total_loan', 'year_of_loan', 'interest', 'monthly_payment',
                         'debt_loan_ratio', 'del_in_18month', 'scoring_low', 'scoring_high',
                         'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup', 'recircle_b',
                         'recircle_u', 'f0', 'f1', 'f2', 'f3', 'f4', 'early_return', 'early_return_amount',
                         'early_return_amount_3mon']
        self.un_num_item = ['class', 'employer_type', 'industry', 'work_year', 'house_exist', 
                            'censor_status', 'use', 'initial_list_status', 'app_type', 'policy_code']
        self.un_use_item = ['loan_id', 'user_id', 'issue_date', 'post_code', 'region', 'earlies_credit_mon', 'title']
        self.un_num_item_list = {item: list(set(st_df[item].dropna().unique())) for item in self.un_num_item}
        self.mode = mode

    def __getitem__(self, index):
        data = [(0 if pd.isnull(self.df[item][index]) else (self.df[item][index] - self.mean_df[item]) / self.std_df[item]) 
                for item in self.num_item]
        emb_data = [self.un_num_item_list[item].index(self.df[item][index]) 
                    if self.df[item][index] in self.un_num_item_list[item] else -1 
                    for item in self.un_num_item]
        data.extend(emb_data)
        loan_id = self.df['loan_id'][index]
        return np.array(data, dtype=np.float32), loan_id  # 直接返回整数ID，无需更改

    def __len__(self):
        return len(self.df)

# 加载模型
model = joblib.load('./SVM/model/svm_model.joblib')

# 初始化测试数据集
test_dataset = MyDataset('./SVM/data/test_public.csv')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

# 预测并保存结果
results = []
for data, loan_id in test_loader:
    prob = model.predict_proba(data)[:, 1]  # 获取正类的概率值
    results.append([loan_id.item(), prob[0]])  # 使用 .item() 方法获取ID的数值

# # 对结果进行处理
# results = sorted(results, key=lambda x: x[1])  # 按概率排序
# n = len(results)
# top_10_percent_index = int(0.85 * n)
# bottom_40_percent_index = int(0.6 * n)
# bottom_80_percent_index = int(0.8 * n)

# for i in range(n):
#     if i < bottom_40_percent_index:
#         results[i][1] = 0
#     elif i < bottom_80_percent_index:
#         results[i][1] *= 0.2
#     elif i < top_10_percent_index:
#         results[i][1] *= 2
#     else:
#         results[i][1] = 1

# 保存结果到CSV文件
pd.DataFrame(results, columns=['id', 'isDefault']).to_csv('./SVM/output/submission.csv', index=False)
