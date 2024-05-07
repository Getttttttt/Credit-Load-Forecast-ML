import joblib
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from deap import base, creator, tools, algorithms, gp
import operator

class MyDataset(Dataset):
    def __init__(self, csv_dir, standard_csv_dir='./GeneticAlgorithm/data/train_public.csv', mode='train'):
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

# 定义适应度评估函数
def evalGA(individual):
    func = toolbox.compile(expr=individual)
    predictions = np.array([func(*record) for record in X_train], dtype=np.float64)
    # 检查 predictions 是否有有效值，防止 NaN
    if np.all(np.isnan(predictions)) or np.var(predictions) == 0:
        return (0.5,)  # 如果预测结果没有变化或全部为 NaN，返回中性的 AUC 分数

    # 检查是否存在 NaN，并在需要时进行处理
    if np.isnan(predictions).any():
        predictions = np.nan_to_num(predictions)  # 将 NaN 替换为数值（通常为 0）

    # 确保不会除以零，执行安全的归一化
    range_predictions = predictions.max() - predictions.min()
    if range_predictions == 0:
        return (0.5,)  # 所有值相同，返回中性的 AUC 分数

    predictions = (predictions - predictions.min()) / range_predictions
    auc = roc_auc_score(y_train, predictions)
    return (auc,)


# DataLoader设置
train_dataset = MyDataset('./GeneticAlgorithm/data/train_public.csv')
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, drop_last=True)

# 准备数据
X_train = []
y_train = []
for data, label in train_loader:
    X_train.append(data)
    y_train.append(label)

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# 定义遗传程序的基本设置
pset = gp.PrimitiveSet("MAIN", arity=len(X_train[0]))
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(np.negative, 1)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

# 定义适应度函数和个体
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalGA)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# 定义适应度评估函数
def evalGA(individual):
    func = toolbox.compile(expr=individual)
    predictions = np.array([func(*record) for record in X_train], dtype=np.float64)
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())  # 归一化
    auc = roc_auc_score(y_train, predictions)
    return (auc,)

# 运行遗传算法
population = toolbox.population(n=300)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=40, verbose=True)

# 找出最好的个体
top_individual = tools.selBest(population, k=1)[0]
print("Best Individual: ", top_individual)
print("Best AUC: ", top_individual.fitness.values[0])

# 保存模型和阈值决策向量
model = {"scaler": scaler, "program": top_individual}
joblib.dump(model, './GeneticAlgorithm/model/genetic_program_model.joblib')
