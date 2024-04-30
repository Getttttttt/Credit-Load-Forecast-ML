import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from joblib import dump
import time
import numpy as np

'''
solver = 'liblinear'
l1 
c = 1000 0.7854 0.7939
c = 100/250/500/750 AUC 0.7854 0.7939
c= 10/25/50/75 AUC:0.7854 0.7939
c = 5/7 0.7854 0.7939
C = 1 AUC:0.7854  0.7939
c = 0.5/0.7 0.7854  0.7939
c = 0.3 0.7854 0.7939
c = 0.1 AUC 0.7854 0.7939

c = 0.01 AUC 0.7851 0.7927

l2
小于0.78

solver = 'sag'
l2
0.6,小于0.7

solver = 'saga'
l2
小于0.78

'''

# 加载数据
data_path = "D:\AI&DL飞浆\\train_done.csv"
df = pd.read_csv(data_path, encoding='gbk')

X = df.drop(columns=['loan_id', 'user_id', 'is_default'])
y = df['is_default']
N = 100

logistic_classifier = LogisticRegression(
        solver='liblinear',
        random_state=3407,
        penalty='l1',
        C=1,
        max_iter=1000)

cv = StratifiedKFold(n_splits=N, shuffle=True, random_state=3407)

auc_scores = []
cross_entropy_losses = []
train_scores = []
test_scores = []
start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    fold_start = time.time()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logistic_classifier.fit(X_train, y_train)

    y_pred = logistic_classifier.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    auc_scores.append(auc_score)
    cross_entropy_loss = log_loss(y_test, y_pred)
    cross_entropy_losses.append(cross_entropy_loss)
    train_score = logistic_classifier.score(X_train, y_train)
    test_score = logistic_classifier.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    fold_end = time.time()
    elapsed = fold_end - fold_start
    estimated = (fold_end - start_time) / (fold + 1) * (N - (fold + 1))
    print('---------------------------------------------------------------')
    print(f'Fold {fold + 1}/{N}, AUC: {auc_score:.4f}, CEL:{cross_entropy_loss:.4f},\n'
            f'train_score:{train_score:.2f},test_score:{test_score}\n'
            f'Elapsed time for this fold: {elapsed:.2f} s, '
            f'Estimated time remaining: {estimated:.2f} s')

average_auc = np.mean(auc_scores)
average_cross_entropy_loss = np.mean(cross_entropy_losses)
average_train_score = np.mean(train_scores)
average_test_score = np.mean(test_scores)
print('*******************************************')
print(f'average AUC: {average_auc}')
print(f'average cross entropy loss: {average_cross_entropy_loss}')
print(f'average_train_score:{average_train_score}')
print(f'average_test_score:{average_test_score}')
print('*******************************************')
with open('D:\AI&DL飞浆\\output.txt', 'w', encoding='utf-8') as result_file:
    result_file.write(f'平均AUC分数: {average_auc}\n')
    result_file.write(f'平均交叉熵损失: {average_cross_entropy_loss}')
    result_file.write(f'训练集上模型平均得分:{average_train_score}')
    result_file.write(f'测试集上模型平均得分:{average_test_score}')

logistic_classifier.fit(X, y)  # 训练最终模型
model_filename = 'D:\AI&DL飞浆\\logistic_regression_model.joblib'
dump(logistic_classifier, model_filename)
print(f'save model is: {model_filename}')