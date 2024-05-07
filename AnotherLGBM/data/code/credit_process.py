#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')

import os
import re
import gc

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)

from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


# # 读取数据

# In[3]:


train_data = pd.read_csv('../raw_data/train_public.csv')

train_data['target'] = 1

print(train_data.shape)
train_data.head()


# In[4]:


train_internet = pd.read_csv('../raw_data/train_internet.csv')

train_internet['target'] = 0

print(train_internet.shape)
train_internet.head()


# In[5]:


train_data['isDefault'].value_counts(dropna=True)


# In[6]:


train_internet = train_internet.rename(columns={'is_default': 'isDefault'})
train_internet['isDefault'].value_counts(dropna=True)


# # 数据整理

# In[7]:


drop1 = ['sub_class', 'work_type', 'house_loan_status', 'marriage', 'offsprings', 'f5']
drop2 = ['known_outstanding_loan', 'known_dero', 'app_type']

train_internet.drop(drop1 + ['user_id'], axis=1, inplace=True)
train_data.drop(drop2 + ['user_id'], axis=1, inplace=True)

train_data = pd.concat([train_data, train_internet]).reset_index(drop=True)
print(train_data.shape)
train_data.head()


# In[8]:


# data = pd.concat([train_data, test_data])
data = train_data.copy()

print(data.shape)
data.head()


# In[9]:


data['issue_date'] = pd.to_datetime(data['issue_date'])
data['issue_mon'] = data['issue_date'].dt.year * 100 + data['issue_date'].dt.month
data.drop(['issue_date'], axis=1, inplace=True)


# In[10]:


data['class'] = data['class'].map({
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'E': 4, 'F': 5, 'G': 6
})


# In[11]:


lbe = LabelEncoder()
data['employer_type'] = lbe.fit_transform(data['employer_type'])


# In[12]:


lbe = LabelEncoder()
data['industry'] = lbe.fit_transform(data['industry'])


# In[13]:


data['work_year'] = data['work_year'].map({
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
})

data['work_year'].fillna(-1, inplace=True)


# In[14]:


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

data['earlies_credit_mon'] = data['earlies_credit_mon'].apply(lambda x: clean_mon(x))


# In[15]:


data.head()


# # 模型

# In[16]:


train = data.copy()
test  = data[data['target'] == 0].copy()

ycol = 'target'
feature_names = list(
    filter(lambda x: x not in [ycol, 'loan_id', 'policy_code'], train.columns))

model = lgb.LGBMClassifier(objective='binary',
                           boosting_type='gbdt',
                           tree_learner='serial',
                           num_leaves=32,
                           max_depth=6,
                           learning_rate=0.1,
                           n_estimators=10000,
                           subsample=0.8,
                           feature_fraction=0.6,
                           reg_alpha=0.5,
                           reg_lambda=0.5,
                           random_state=2021,
                           is_unbalance=True,
                           metric='auc')


oof = []
prediction = test[['loan_id']]
prediction[ycol] = 0
df_importance_list = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):
    X_train = train.iloc[trn_idx][feature_names]
    Y_train = train.iloc[trn_idx][ycol]

    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][ycol]

    print('\nFold_{} Training ================================\n'.format(fold_id+1))

    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric='auc',
                          early_stopping_rounds=50)

    pred_val = lgb_model.predict_proba(
        X_val, num_iteration=lgb_model.best_iteration_)
    df_oof = train.iloc[val_idx][['loan_id', ycol]].copy()
    df_oof['pred'] = pred_val[:, 1]
    oof.append(df_oof)

    pred_test = lgb_model.predict_proba(
        test[feature_names], num_iteration=lgb_model.best_iteration_)
    prediction[ycol] += pred_test[:, 1] / kfold.n_splits

    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
    gc.collect()
    


# In[17]:


df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()
df_importance


# In[18]:


oof = pd.concat(oof)
print('roc_auc_score:', roc_auc_score(oof['target'], oof['pred']))


# In[19]:


prediction.info()


# In[20]:


prediction['target'].describe()


# In[21]:


save_list = prediction[prediction['target'] >= 0.1]['loan_id'].tolist()
len(save_list)


# In[22]:


train_internet = pd.read_csv('../raw_data/train_internet.csv')
train_internet = train_internet[train_internet['loan_id'].isin(save_list)].copy()
print(train_internet.shape)
train_internet.head()


# In[23]:


train_internet = train_internet.rename(columns={'is_default': 'isDefault'})
train_internet['isDefault'].value_counts()


# In[24]:


train_internet[['total_loan']]=train_internet[['total_loan']].astype(np.int64)


# In[25]:


train_internet.to_csv(f'../user_data/train_internet_{len(train_internet)}.csv', index=False)


# In[ ]:





# In[ ]:




