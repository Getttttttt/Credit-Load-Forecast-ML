

# In[1]:   


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# In[1]:

# Load data
train_data = pd.read_csv('./ReplacePreprocessDataForSVM/raw_data/nw_train_public.csv')
test_public = pd.read_csv('./ReplacePreprocessDataForSVM/raw_data/test_public.csv')
train_inte = pd.read_csv('./ReplacePreprocessDataForSVM/raw_data/train_internet.csv')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Utility functions
def workYearDIc(x):
    if str(x) == 'nan':
        return -1
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())

def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'

# Data preprocessing
class_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
timeMax = pd.to_datetime('1-Dec-21')
for dataset in [train_data, test_public, train_inte]:
    dataset['work_year'] = dataset['work_year'].map(workYearDIc)
    dataset['class'] = dataset['class'].map(class_dict)
    dataset['earlies_credit_mon'] = pd.to_datetime(dataset['earlies_credit_mon'].map(findDig))
    dataset.loc[dataset['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] += pd.offsets.DateOffset(years=-100)
    dataset['issue_date'] = pd.to_datetime(dataset['issue_date'])
    dataset['issue_date_month'] = dataset['issue_date'].dt.month
    dataset['issue_date_dayofweek'] = dataset['issue_date'].dt.dayofweek
    dataset['earliesCreditMon'] = dataset['earlies_credit_mon'].dt.month
    dataset['earliesCreditYear'] = dataset['earlies_credit_mon'].dt.year

cat_cols = ['employer_type', 'industry']
for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    for dataset in [train_data, test_public, train_inte]:
        dataset[col] = lbl.transform(dataset[col])

col_to_drop = ['issue_date', 'earlies_credit_mon']
for dataset in [train_data, test_public, train_inte]:
    dataset.drop(col_to_drop, axis=1, inplace=True)

# Merge data
y = train_data['isDefault']
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)

def train_model(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')  # or median, most_frequent depending on data characteristics
    
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    
    # Apply imputation
    data_imputed = imputer.fit_transform(data_[feats])
    test_imputed = imputer.transform(test_[feats])
    
    # Apply scaling
    data_scaled = scaler.fit_transform(data_imputed)
    test_scaled = scaler.transform(test_imputed)

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_scaled[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_scaled[val_idx], y_.iloc[val_idx]
        
        clf = SVC(probability=True, kernel='rbf', C=1.0, verbose=True)
        clf.fit(trn_x, trn_y)
        
        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
        sub_preds += clf.predict_proba(test_scaled)[:, 1] / folds_.n_splits
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds)) 
    test_['isDefault'] = sub_preds

    return oof_preds, test_[['id', 'isDefault']]

oof_preds, test_preds = train_model(train_data, test_public, y, folds)
test_preds.to_csv('svm_predictions.csv', index=False)

# %%
