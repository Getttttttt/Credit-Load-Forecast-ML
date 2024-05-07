import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class MyDataset(Dataset):
    def __init__(self, csv_dir, standard_csv_dir='./SVM/data/train_public.csv', mode='train'):
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
        label = self.df['isDefault'][index] if self.mode == 'train' else self.df['loan_id'][index]
        return np.array(data, dtype=np.float32), label

    def __len__(self):
        return len(self.df)

# DataLoader设置
train_dataset = MyDataset('./SVM/data/train_public.csv')
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, drop_last=True)

# 准备数据
X_train = []
y_train = []
for data, label in train_loader:
    X_train.append(data)
    y_train.append(label)

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

# 使用管道整合标准化和SVM模型
model = make_pipeline(StandardScaler(), SVC(probability=True, class_weight={1: 5}))

# 训练模型
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, './SVM/model/svm_model.joblib')
