import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from deap import gp, creator, base
import operator
import random
from torch.utils.data import DataLoader, Dataset

# 为遗传算法定义操作集
pset = gp.PrimitiveSet("MAIN", arity=31)  # 修改此处以匹配特征的数量
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(np.negative, 1)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 加载模型
model = joblib.load('./GeneticAlgorithm/model/genetic_program_model.joblib')
scaler = model['scaler']
program = model['program']

# 定义评估函数
def evalGA(individual, X):
    func = gp.compile(expr=individual, pset=pset)
    predictions = np.array([func(*record) for record in X], dtype=np.float64)
    return predictions

# 测试数据集类，与训练数据集类似，只是去除了标签的部分
class TestDataset(Dataset):
    def __init__(self, csv_dir, standard_csv_dir='./GeneticAlgorithm/data/train_public.csv'):
        super(TestDataset, self).__init__()
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

    def __getitem__(self, index):
        data = [(0 if pd.isnull(self.df[item][index]) else (self.df[item][index] - self.mean_df[item]) / self.std_df[item]) 
                for item in self.num_item]
        emb_data = [self.un_num_item_list[item].index(self.df[item][index]) 
                    if self.df[item][index] in self.un_num_item_list[item] else -1 
                    for item in self.un_num_item]
        data.extend(emb_data)
        loan_id = self.df['loan_id'][index]
        return np.array(data, dtype=np.float32), loan_id

    def __len__(self):
        return len(self.df)

# 测试数据加载
test_dataset = TestDataset('./GeneticAlgorithm/data/test_public.csv')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 预测
predictions = []
for data, loan_id in test_loader:
    data = scaler.transform(data)  # 使用训练时的scaler标准化数据
    prediction = evalGA(program, data)
    predictions.append([loan_id.item(), prediction[0]])

# 可选：输出预测结果，或者保存到文件
pd.DataFrame(predictions, columns=['id', 'isDefault']).to_csv('./GeneticAlgorithm/output/submission.csv', index=False)
