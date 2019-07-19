"""
支持向量机模型的建立
"""

# 常用科学计算包
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import load_breast_cancer
# sklearn中的训练集与数据集分离包
from sklearn.model_selection import train_test_split
# 支持向量机算法包
from sklearn.svm import SVC

# 数据实例化
cancer = load_breast_cancer()

# 测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# 支持向量机模型建立
svc = SVC()
svc.fit(X_train, y_train)

# 模型评估
print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

# 数据分散程度显示
plt.figure()
plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")


# 数据预处理（对数据进行缩放：所有数据处于[0-1]）
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
print("Maximum for each feature\n{}".format(X_train_scaled.max(axis=0)))
# 预处理后的数据分散显示
plt.figure()
plt.plot(X_train_scaled.min(axis=0), 'o', label="min")
plt.plot(X_train_scaled.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.show()

# 对预处理后的数据进行建模
svc0 = SVC()
svc0.fit(X_train_scaled, y_train)

# 模型评估
print("Accuracy on training set0: {:.3f}".format(svc0.score(X_train_scaled, y_train)))
print("Accuracy on test set0: {:.3f}".format(svc0.score(X_test_scaled, y_test)))

"""
2019.7.11
学习总结：
    1.采用支持向量机构建模型对乳腺癌判定进行预测，有两个重要参数：核宽度gamma、正则化参数C
    该算法对数据要求较高，需要进行数据预处理，使其分布较为紧凑，不能相差太多
    2.matplotlib.pyplot类的使用：
    # plot作图：对于一维数据，输入为纵坐标表示数据点数值大小，横坐标自动生成
    # 而且可以对数据点进行分类：label属性（牢记）
    plt.plot(X_train.min(axis=0), 'o', label="min")
    plt.plot(X_train.max(axis=0), '^', label="max")
    # 种类显示（loc = 放置位置）
    plt.legend(loc=4)
    # y轴格式（"对数形式"）
    plt.yscale("log")
    # 弹出画图窗口（show）
    plt.show()
    3.熟悉机器学习的基本流程
需要回忆：
    对书中出现的数据可视化代码进行查找，并进一步改写此代码
    有必要对最近集中数据可视化方式进行一个详细的总结
    python基本数据类型基础：列表（list）、数组（numpy.array）、矩阵（DateFrame）
理论总结：
    线性支持向量机是支持向量机的一种特殊形式
    对于线性模型，当特征数所构造的线性函数不能很好的进行分类的时候，可以将其特征的组合项（非线性特征）作为其特征之一，进行线性模型的构建
    根据如何选择组合项发展出了支持向量机的方法
"""