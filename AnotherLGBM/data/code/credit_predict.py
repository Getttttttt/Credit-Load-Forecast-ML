#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 统一导入工具包及显示设置
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import StratifiedKFold
from dateutil.relativedelta import relativedelta
# 显示设置
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)
import warnings
warnings.simplefilter('ignore')


# In[2]:


# 读入数据
train_data = pd.read_csv('../raw_data/train_public.csv')
train1_data = pd.read_csv('../user_data/train_internet_4248.csv')
test_public = pd.read_csv('../raw_data/test_public.csv')
train_inte = pd.read_csv('../raw_data/train_internet.csv')
train_inte = train_inte.rename(columns={'is_default': 'isDefault'})
train_inte[['total_loan']]=train_inte[['total_loan']].astype(np.int64)


# In[7]:


# 处理earlies_credit_mon函数
def clean_mon(x):
    mons = {'jan':1, 'feb':2, 'mar':3, 'apr':4,  'may':5,  'jun':6,
            'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    year_group = re.search('(\d+)', x)
    if year_group:
        year = int(year_group.group(1))
        if year < 22:
            year += 2000
        elif 100 > year > 22:
            year += 1900
        else:
            pass
    else:
        year = 2022
        
    month_group = re.search('([a-zA-Z]+)', x)
    if month_group:
        mon = month_group.group(1).lower()
        month = mons[mon]
    else:
        month = 0 
    return year*100 + month

# 统一处理日期
def process_date(data):
    data['issue_date'] = pd.to_datetime(data['issue_date'])
    data['issue_mon'] = data['issue_date'].dt.year * 100 + data['issue_date'].dt.month
    data['issue_date_dayofweek'] = data['issue_date'].dt.dayofweek
    data['earlies_credit_mon'] = data['earlies_credit_mon'].apply(lambda x: clean_mon(x))
    col_to_drop = ['issue_date','policy_code']
    data.drop(col_to_drop,axis=1,inplace=True)
    # 增加一个表示从开卡到这次发放贷款的间隔时间
    data['CreditLine'] = data['issue_mon']-data['earlies_credit_mon']
    return data

train_data = process_date(train_data)
test_public= process_date(test_public)
train_inte = process_date(train_inte)
train1_data = process_date(train1_data)


# In[4]:


train_data.head(5)


# In[8]:


# 处理工作年数，类别数据等
from sklearn.preprocessing import LabelEncoder
def workYearDIc(x):
    if str(x) == 'nan':
        return -1
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())

class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}

# 处理年份及各种特征工程
def process_year_cat(data):
    data['work_year'] = data['work_year'].map(workYearDIc)
    data['class'] = data['class'].map(class_dict)
    
#     增加一个表示客户评分平均分的属性
    data['scoring'] = (data['scoring_high']+data['scoring_low'])/2
    
#     清理错误数据，提前还款总额不为0，还款次数为0的置为1
    data.loc[(data['early_return']==0)&(data['early_return_amount']>0),['early_return']]=1
    
#     增加一个表示是否有提前还款的特征
    data.loc[data['early_return']>=1,['early_return_YN']]=1
    data.loc[data['early_return']==0,['early_return_YN']]=0
    
#     增加一个贷款类别是否为A的特征   
    data.loc[data['class']==1,['class_A']]=1
    data.loc[data['class']!=1,['class_A']]=0
    
#     增加一个表示是否有房的特征： 实际效果有所下降
    data.loc[data['house_exist']>=1,['house_YN']] = 1
    data.loc[data['house_exist']==0,['house_YN']] = 0

#     贷款金额/分期付款
    data['totloan_monpay'] = data['total_loan']/data['monthly_payment']
    
#     增加一个表示借款人是否在过去18个月逾期30天以上的违约
    data.loc[data['del_in_18month']>=1,['del_in_18month_YN']]=1
    data.loc[data['del_in_18month']==0,['del_in_18month_YN']]=0

#     增加一个是否清除公共记录的数量
    data.loc[data['pub_dero_bankrup']>=1,['pub_dero_bankrup_YN']]=1
    data.loc[data['pub_dero_bankrup']==0,['pub_dero_bankrup_YN']]=0
    cat_cols = ['employer_type','industry']
    for col in cat_cols:
        lbl = LabelEncoder().fit(data[col])
        data[col] = lbl.transform(data[col])
    return data

train_data = process_year_cat(train_data)
test_public = process_year_cat(test_public)
train_inte = process_year_cat(train_inte)
train1_data = process_year_cat(train1_data)


# In[6]:


# 查找关联
corr_matrix = train_data.corr()
corr_matrix["isDefault"].sort_values(ascending=False)


# In[7]:


#  train_cols: train_data中的属性， same_cols: 共同属性
train_cols = set(train_data.columns)
same_cols = list(train_cols.intersection(set(train_inte.columns)))
train_inteSame = train_inte[same_cols].copy()
train1_dataSame = train1_data[same_cols].copy()
#  给internet数据添加与public一样的属性列，并设值为nan
Inte_add_cols = list(train_cols.difference(set(same_cols)))
for col in Inte_add_cols:
    train_inteSame[col] = np.nan
    train1_dataSame[col] = np.nan

train_data =  pd.concat([train_data,train1_dataSame]).reset_index(drop=True)

# 使用KNN填补数据
from sklearn.impute import KNNImputer
imputer  = KNNImputer(n_neighbors=10)
train_data_filled = imputer.fit_transform(train_data)
train_data = pd.DataFrame(train_data_filled,columns=train_data.columns)


# In[8]:


train_data.shape


# In[11]:


# 模型
import catboost as cb
from sklearn.model_selection import GridSearchCV
def train_model(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_[feats],data_['isDefault'])):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
        
        clf = cb.CatBoostClassifier(iterations=10000, 
                                      depth=7, 
                                      learning_rate=0.002, 
                                      loss_function='Logloss',
                                      eval_metric='AUC',
                                      logging_level='Verbose', 
                                      metric_period=50)
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)])
# ,verbose=50,early_stopping_rounds=200
        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
        sub_preds += clf.predict_proba(test_[feats])[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['isDefault'] = sub_preds

    return oof_preds, test_[['loan_id', 'isDefault']], feature_importance_df


def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


# In[13]:


y = train_data['isDefault']
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=202110)

# IntePre：返回test数据的id及对应预测结果
oof_preds, IntePre, importances = train_model(train_data, train_inteSame, y, folds)


# In[ ]:





# In[14]:


IntePre['isDef'] = train_inte['isDefault']
from sklearn.metrics import roc_auc_score

roc_auc_score(IntePre['isDef'], IntePre.isDefault)


# In[15]:


IntePre.describe()


# In[16]:


# 选择阈值0.05，从internet表中提取预测小于该概率的样本，并对不同来源的样本赋予来源值
InteId1 = IntePre.loc[(IntePre.isDefault < 0.05), 'loan_id'].tolist()
InteId2 = IntePre.loc[(IntePre.isDefault < 0.10)&(IntePre.isDefault>0.03), 'loan_id'].tolist()


train_data['dataSourse'] = 1
test_public['dataSourse'] = 1
train_inteSame['dataSourse'] = 0
train_inteSame['isDefault'] = train_inte['isDefault']
use_te1= train_inteSame[train_inteSame.loan_id.isin(InteId1)].copy()
use_te2= train_inteSame[train_inteSame.loan_id.isin(InteId2)].copy()
data1 = pd.concat([train_data, test_public, use_te1]).reset_index(drop=True)
data2 = pd.concat([train_data, test_public, use_te2]).reset_index(drop=True)
len(InteId1)


# In[17]:


len(InteId2)


# In[18]:


# IntePre.isDefault
plt.figure(figsize=(16, 6))
plt.title("Distribution of Default values IntePre")
sns.distplot(IntePre['isDefault'], color="black", kde=True, bins=120, label='train_data')


# In[19]:


train1 = data1[data1['isDefault'].notna()]
train2 = data2[data2['isDefault'].notna()]
test = data1[data1['isDefault'].isna()]
print(len(train1))
print(len(train2))
# 填充test数据
# 使用KNN填补数据
from sklearn.impute import KNNImputer
imputer  = KNNImputer(n_neighbors=10)
test_data_filled = imputer.fit_transform(test)
imputer_columns=[]
for col in test.columns:
    if(col!='isDefault'):
        imputer_columns.append(col)

test = pd.DataFrame(test_data_filled,columns=imputer_columns)
#将测试集划分为是否提前还款
testId1 = test.loc[(test.early_return != 0), 'loan_id'].tolist()
testId2 = test.loc[(test.early_return == 0), 'loan_id'].tolist()
test1= test[test.loan_id.isin(testId1)].copy()
test2= test[test.loan_id.isin(testId2)].copy()

y1 = train1['isDefault']
print(len(y1))
print(len(train1))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=202110)
oof_preds1, test_preds1, importances1 = train_model(train1, test1, y1, folds)
test_preds1.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('../user_data/nn0.csv', index=False)


y2 = train2['isDefault'].copy()
print(len(y2))
print(len(train2))

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=202110)
oof_preds2, test_preds2, importances2 = train_model(train2, test2, y2, folds)
test_preds2.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('../user_data/nn1.csv', index=False)

test_preds= pd.concat([test_preds1, test_preds2]).reset_index(drop=True)
test_preds.to_csv('../user_data/nn2.csv', index=False) 


# In[ ]:





# In[20]:


test.info()
print(test_data_filled.shape)


# In[21]:


# 将预测的值小于0.5的isDefault值设为0，并保存到一文件中，扩大训练集
train_data = pd.read_csv('../raw_data/train_public.csv')
test_data = pd.read_csv('../raw_data/test_public.csv')
sub = pd.read_csv("../user_data/nn2.csv")
sub = sub.rename(columns={'id': 'loan_id'})
sub.loc[sub['isDefault'] < 0.50, 'isDefault'] = 0
sub.loc[sub['isDefault'] > 0.68, 'isDefault'] = 1
nw_sub0 = sub[(sub['isDefault'] == 0)]
nw_sub1 = sub[(sub['isDefault'] == 1)]
nw_test_data0 = test_data.merge(nw_sub0, on='loan_id', how='inner')
nw_test_data1 = test_data.merge(nw_sub1, on='loan_id', how='inner')
nw_train_data = pd.concat([train_data, nw_test_data0,nw_test_data1]).reset_index(drop=True)
nw_train_data.to_csv("../user_data/nw_train_public.csv", index=0)
nw_test_data1.shape


# In[22]:


nw_test_data0


# In[23]:


train_data.head()


# In[5]:


train_data = pd.read_csv('../user_data/nw_train_public.csv')
train1_data = pd.read_csv("../user_data/train_internet_4248.csv")
test_public = pd.read_csv('../raw_data/test_public.csv')
train_inte = pd.read_csv('../raw_data/train_internet.csv')
train_inte = train_inte.rename(columns={'is_default': 'isDefault'})
train_inte[['total_loan']]=train_inte[['total_loan']].astype(np.int64)


# In[6]:


# 处理datetime数据
train_data = process_date(train_data)
test_public= process_date(test_public)
train_inte = process_date(train_inte)
train1_data = process_date(train1_data)

# 处理年数和类别数据
train_data = process_year_cat(train_data)
test_public = process_year_cat(test_public)
train_inte = process_year_cat(train_inte)
train1_data = process_year_cat(train1_data)


# In[9]:


# train_cols: train_data中的属性， same_cols: 共同属性
train_cols = set(train_data.columns)
same_cols = list(train_cols.intersection(set(train_inte.columns)))
train_inteSame = train_inte[same_cols].copy()
train1_dataSame = train1_data[same_cols].copy()
# 给internet数据添加与public一样的属性列，并设值为nan
Inte_add_cols = list(train_cols.difference(set(same_cols)))
for col in Inte_add_cols:
    train_inteSame[col] = np.nan
    train1_dataSame[col] = np.nan

train_data =  pd.concat([train_data,train1_dataSame]).reset_index(drop=True)

# 使用KNN填补数据
from sklearn.impute import KNNImputer
imputer  = KNNImputer(n_neighbors=10)
train_data_filled = imputer.fit_transform(train_data)
train_data = pd.DataFrame(train_data_filled,columns=train_data.columns)


# In[ ]:





# In[12]:


y = train_data['isDefault']
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=202110)

# IntePre：返回test数据的id及对应预测结果
oof_preds, IntePre, importances = train_model(train_data, train_inteSame, y, folds)


# In[13]:


IntePre['isDef'] = train_inte['isDefault']
from sklearn.metrics import roc_auc_score

roc_auc_score(IntePre['isDef'], IntePre.isDefault)


# In[14]:


InteId1 = IntePre.loc[(IntePre.isDefault < 0.05), 'loan_id'].tolist()
InteId2 = IntePre.loc[(IntePre.isDefault < 0.10)&(IntePre.isDefault>0.03), 'loan_id'].tolist()


train_data['dataSourse'] = 1
test_public['dataSourse'] = 1
train_inteSame['dataSourse'] = 0
train_inteSame['isDefault'] = train_inte['isDefault']
use_te1= train_inteSame[train_inteSame.loan_id.isin(InteId1)].copy()
use_te2= train_inteSame[train_inteSame.loan_id.isin(InteId2)].copy()
data1 = pd.concat([train_data, test_public, use_te1]).reset_index(drop=True)
data2 = pd.concat([train_data, test_public, use_te2]).reset_index(drop=True)


# In[15]:


# IntePre.isDefault
plt.figure(figsize=(16, 6))
plt.title("Distribution of Default values IntePre")
sns.distplot(IntePre['isDefault'], color="black", kde=True, bins=120, label='train_data')


# In[ ]:


train1 = data1[data1['isDefault'].notna()]
train2 = data2[data2['isDefault'].notna()]
test = data1[data1['isDefault'].isna()]


testId1 = test.loc[(test.early_return != 0), 'loan_id'].tolist()
testId2 = test.loc[(test.early_return == 0), 'loan_id'].tolist()
test1= test[test.loan_id.isin(testId1)].copy()
test2= test[test.loan_id.isin(testId2)].copy()
print(test1.info())


y1 = train1['isDefault']
print(len(y1))
print(len(train1))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=202110)
oof_preds1, test_preds1, importances1 = train_model(train1, test1, y1, folds)
test_preds1=test_preds1.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']]


y2 = train2['isDefault'].copy()
print(len(y2))
print(len(train2))

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=202110)
oof_preds2, test_preds2, importances2 = train_model(train2, test2, y2, folds)
test_preds2=test_preds2.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']]

test_preds= pd.concat([test_preds1, test_preds2]).reset_index(drop=True)
test_preds.to_csv('../prediction_result/result.csv', index=False)


# In[ ]:


display_importances(importances1)


# In[ ]:


display_importances(importances2)

