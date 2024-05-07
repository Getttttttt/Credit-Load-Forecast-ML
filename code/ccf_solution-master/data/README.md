# 2021CCF个贷违约预测赛道解决方案

## 环境

python 3.7

##  安装教程

```
pip install tqdm
pip install runipy
pip install scikit-learn
pip install python-dateutil
pip install catboost
pip install matplotlib
pip install seaborn
pip install pandas
pip install lightgbm
pip install numpy
```

## 解决方案

分为两个步骤，分别在文件credit_process.py， credit_predict.py中进行

### credit_process.py

通过「对抗验证」从 train_internet 表中找出部分较为符合 train_public 表分布的数据 ,步骤如下

-  将train_public数据的target 设置为 1，将train_internet 的 target 设置为 0，合并两个表
-  选择lgbm模型，训练模型分辨两个表的数据  
- 将训练好的模型预测概率较高的internet数据挑选出来，并保存到user_data/train_internet_4248.csv中

### credit_predict.py

该过程利用credit_process.py找出的数据集扩大训练集进行预测，具体步骤如下：

- 分别读入train_public, train_internet, test_public, train_internet_4248数据
- 对统一的数据预处理，包括日期处理，文本信息映射
- 对数据做特征工程：错误信息处理，特征提取与特征组合，KNN数据填充
- 合并train_public 和 train_internet_4248 数据作为训练集new_train_public
- 模型选择：选择 catboost，通过5折交叉验证
- 数据划分:将test_public 通过有无提前还款划分成两个测试集test_1,test_2
- 多层数据增强
  1、  使用新训练集对模型进行训练，然后预测train_internet数据，提取出两个特定范围内的数据(参数<0.05的为 internet_1,参数为0.03~0.10的为internet_2)，分别将internet_1，internet_2添加到新训练集中形成new_train_public_1,new_train_public_2训练集,进行模型训练后对test_1，test_2进行预测，将两个测试集预测结果合并形成test_public。
  2、  提取第1步测试集结果中特定范围的数据，与new_train_public合并作为新的训练集。
  3、  重复第1步，对测试集进行最终预测。

## 文件结构

```
|-- data
  |-- code
    |-- credit_predict.py
    |-- credit_process.py
    |-- main.py
  |-- prediction_result
  |-- raw_data
    |-- train_public.csv
    |-- train_internet.csv
    |-- test_public.csv
  |-- user_data   
```

## 使用说明

```
python code/main.py
```


- 使用者只需运行 main.py 文件即可运行整个代码，该过程会在 use_data 目录下生成中间文件，预测结果文件生成于prediction_result 目录下的 result.csv 文件
- 该过程大概需要2-3个小时，具体因机器的配置而异

  