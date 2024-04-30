import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    # csv_dir对应要读取的数据地址，standard_csv_dir用于生成均值和方差信息对数据进行归一化的文件地址
    def __init__(self,csv_dir,standard_csv_dir='./NeuralNetworkExample/data/train_public.csv',mode = 'train'):
        super(MyDataset, self).__init__()

        # 读取数据
        self.df = pd.read_csv(csv_dir)
        
        # 构造各个变量的均值和方差
        st_df = pd.read_csv(standard_csv_dir)
        st_df = st_df.apply(pd.to_numeric, errors='coerce')
        self.mean_df = st_df.mean()
        self.std_df = st_df.std()

        # 分别指定数值型变量/分类变量/不使用的变量
        self.num_item = ['total_loan', 'year_of_loan', 'interest','monthly_payment',
        'debt_loan_ratio', 'del_in_18month', 'scoring_low','scoring_high', 'known_outstanding_loan', 'known_dero','pub_dero_bankrup', 'recircle_b', 'recircle_u', 
        'f0', 'f1','f2', 'f3', 'f4', 'early_return', 'early_return_amount','early_return_amount_3mon']
        self.un_num_item = ['class','employer_type','industry','work_year','house_exist', 'censor_status',
        'use',
        'initial_list_status','app_type',
        'policy_code']
        self.un_use_item = ['loan_id', 'user_id',
        'issue_date', 
        'post_code', 'region',
        'earlies_credit_mon','title']

        # 构造一个映射表，将分类变量/分类字符串映射到对应数值上
        un_num_item_list = {}
        for item in self.un_num_item:
            un_num_item_list[item]=list(set(st_df[item].values))
        self.un_num_item_list = un_num_item_list

        self.mode = mode

    def __getitem__(self, index):
        data=[]

        # 进行归一化，如果这个数值缺省了直接设置为0
        for item in self.num_item:
            if np.isnan(self.df[item][index]):
                data.append((0-self.mean_df[item])/self.std_df[item])
            else:
                data.append((self.df[item][index]-self.mean_df[item])/self.std_df[item])
        
        emb_data = []

        # 将分类变量映射到对应数值上
        for item in self.un_num_item:
            try:
                if self.df[item][index] not in self.un_num_item_list[item]:
                    emb_data.append(-1)
                else:
                    emb_data.append(self.un_num_item_list[item].index(self.df[item][index]))
            except:
                emb_data.append(-1)

        data = torch.tensor(data, dtype=torch.float32)
        emb_data = torch.tensor(emb_data, dtype=torch.float32)

        # 如果当前模式不为train，则返回对应的loan_id，用于锁定样本条目
        if self.mode == 'train':
            label = self.df['isDefault'][index]
        else:
            label = self.df['loan_id'][index]

        label = np.array(label).astype('int64')
        return data,emb_data,label

    def __len__(self):
        return len(self.df)