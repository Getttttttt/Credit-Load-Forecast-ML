import pandas as pd

# 读取CSV文件
df = pd.read_csv('./ReplacePreprocessDataForSVM/Outcome/submission.csv')

# 将isDefault所在列中低于0.5的数值置为0
df['isDefault'] = df['isDefault'].apply(lambda x: 0 if x < 0.5 else x)
df['isDefault'] = df['isDefault'].apply(lambda x: 1 if x > 0.7 else x)

# 将修改后的数据保存回CSV文件
df.to_csv('./ReplacePreprocessDataForSVM/Outcome/submission.csv', index=False)
