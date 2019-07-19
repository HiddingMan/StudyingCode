"""
无监督学习与预处理(两个重要方法：fit、transform)
"""

# 常用科学计算包
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
# sklearn中的训练集与数据集分离包
from sklearn.model_selection import train_test_split
# 数据预处理算法包
from sklearn.preprocessing import MinMaxScaler

# 数据实例化
cancer = load_breast_cancer()
X0, _ = make_blobs()

# 测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# 使用minmaxscaler算法对数据进行缩放
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# 输出缩放之前与缩放之后的数据
print("数据形状:{}".format(X_train_scaled.shape))
print("转换前每个特征的最小值:{}".format(X_train.min(axis=0)))
print("转换前每个特征的最大值:{}".format(X_train.max(axis=0)))
print("转换后每个特征的最小值:{}".format(X_train_scaled.min(axis=0)))
print("转换后每个特征的最大值:{}".format(X_train_scaled.max(axis=0)))


# 对训练数据与测试数据进行相同的缩放结果对比
X0_train, X0_test = train_test_split(X0, random_state=5, test_size=.1)

# 原始训练集和测试集数据可视化
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X0_train[:, 0], X0_train[:, 1], label="training set")
axes[0].scatter(X0_test[:, 0], X0_test[:, 1], label="test set")
axes[0].set_title("original data")
axes[0].legend(loc='upper left')

# 用minmaxscaler算法变换训练集并进行数据可视化
scaler0 = MinMaxScaler()
scaler0.fit(X0_train)
X0_train_scaled = scaler0.transform(X0_train)
X0_test_scaled = scaler0.transform(X0_test)
axes[1].scatter(X0_train_scaled[:, 0], X0_train_scaled[:, 1])
axes[1].scatter(X0_test_scaled[:, 0], X0_test_scaled[:, 1])
axes[1].set_title("scaled data")

# 用minmaxscaler算法对训练机和测试集进行单独变换进行数据可视化
scaler1 = MinMaxScaler()
scaler1.fit(X0_test)
X1_test_scaled = scaler1.transform(X0_test)
axes[2].scatter(X0_train_scaled[:, 0], X0_train_scaled[:, 1])
axes[2].scatter(X1_test_scaled[:, 0], X1_test_scaled[:, 1])
axes[2].set_title("improperly scaled data")
plt.show()

"""
2019.7.14+15+16
学习总结：
    1.本书中两种无监督学习（其弊端即为没有固定输出，不能很好的判断其准确性，不好对算法的好坏进行评估）
    数据集变换：用于创建数据集新的表示方式的一种算法（因为原始数据集有可能更难被人们或者机器学习算法所理解，即特征不明显）
    聚类算法：就是分类，与之前的分类算法类似
    2.四种数据预处理（按特征进行移动和缩放）方法：standardscaler、minmaxscaler、robustscaler、normalizer
    3.对数组数据的调用方式（与matlab类似）：[:, 0]等

需要注意：
    1.对于数据集的无监督变换，尤其重要一点就是要使用训练集所得到的scale进行transform训练集和测试集，禁止单独对测试集进行缩放（数据变换）
    若单独进行变换测试集，则会使数据之间的相对位置放生改变，就破坏了原有数据的相对性（即数据特征被破坏）

回忆数据可视化：创建多个子图并为每个子图添加特征
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # 散点图
    axes[0].scatter(X0_train[:, 0], X0_train[:, 1], label="training set")
    axes[0].scatter(X0_test[:, 0], X0_test[:, 1], label="test set")
    # 设置图标题
    axes[0].set_title("original data")
    # 设置类别图框
    axes[0].legend(loc='upper left')
    
统一总结：
    对于plot、scatter等方法，都具有label属性，添加label属性后，可以设置legend属性
"""

