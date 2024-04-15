import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import dump
import time
import numpy as np
from sklearn.metrics import log_loss


df = pd.read_csv('train_done.csv')

X = df.drop(columns=['loan_id', 'user_id', 'is_default'])
y = df['is_default']
N = 100

# criterion  gini和entropy，默认是gini, entropy本质上惩罚更狠，但是慢一点，而且高噪声数据中很容易过拟合，不一定！！仅仅改变这一点并未提高模型，貌似两个差不多

dt_classifier = DecisionTreeClassifier(random_state=3407,
                                       criterion='entropy',
                                       max_depth = 3,
                                       min_samples_leaf = 50,
                                       min_samples_split = 100,
                                       max_features =  28,
                                       min_impurity_decrease = 0.002
                                       )
'''
    2024-04-15 01:18
    原来无任何参数限制
    train_score:1.00,test_score:0.74  过拟合
    修改参数如下:
    criterion = 'entropy'  # 差别不大
    max_depth = 3  # 效果很好! 2、3、5差别不大
    min_samples_leaf = 5  # 这个没啥用,从5-200都试了
    min_samples_split = 100 # 好像也没啥用
    max_features = 28  # 暴力降维,貌似有奇效,之后尝试一下专门的算法降维PCA,但是效果差不多,没有明显增益
    min_impurity_decrease = 0.002
    ********************************
    调整以后: AUC: 0.73 平均信息熵: 0.44
'''
cv = StratifiedKFold(n_splits=N, shuffle=True, random_state=3407)

auc_scores = []
cross_entropy_losses = []
train_scores=[]
test_scores=[]
start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    fold_start = time.time()
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    dt_classifier.fit(X_train, y_train)
    
    y_pred = dt_classifier.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    auc_scores.append(auc_score)
    y_pred_proba = dt_classifier.predict_proba(X_test)
    cross_entropy_loss = log_loss(y_test, y_pred_proba)
    cross_entropy_losses.append(cross_entropy_loss)
    train_score = dt_classifier.score(X_train,y_train)
    test_score = dt_classifier.score(X_test,y_test)
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
with open('output_2.txt', 'w', encoding='utf-8') as result_file:
    result_file.write( f'平均AUC分数: {average_auc}\n')
    result_file.write( f'平均交叉熵损失: {average_cross_entropy_loss}')
    result_file.write( f'训练集上模型平均得分:{average_train_score}')
    result_file.write( f'测试集上模型平均得分:{average_test_score}')

dt_classifier.fit(X, y)  # 训练最终模型
model_filename = 'decision_tree_model_041502.joblib'
dump(dt_classifier, model_filename)
print(f'save model is: {model_filename}')