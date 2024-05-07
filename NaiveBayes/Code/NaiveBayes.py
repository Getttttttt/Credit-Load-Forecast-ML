import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib
import os

def load_and_combine_data(filepath1, filepath2):
    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# 加载数据
data = load_and_combine_data('./NaiveBayes/Data/train_done_part1.csv', './NaiveBayes/Data/train_done_part2.csv')
print(data.columns)

# 预处理
data = data.drop(data.columns[[0, 1]], axis=1)
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
data = data.fillna(data.mean())

# 分割特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 确保标签是二进制的，适用于二分类问题
y = (y >= 0.5).astype(int)

# 查看类别分布
class_distribution = y.value_counts(normalize=True)
print("Class distribution:", class_distribution)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3407)

# 训练模型
priors = class_distribution.sort_index().values  # 排序并获取类别先验概率
model = GaussianNB(priors=priors)
model.fit(X_train, y_train)

# 预测测试集的概率
y_probs = model.predict_proba(X_test)[:, 1]  # 获取正类的概率

# 计算AUC分数
auc_score = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc_score}")

# 保存模型
model_path = './NaiveBayes/Model/trained_gaussian_nb_model.pth'
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
joblib.dump(model, model_path)
print(f"模型已保存至 {model_path}")


