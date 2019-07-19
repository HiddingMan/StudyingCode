"""
采用k近邻算法对鸢尾花数据进行建模，预测新鸢尾花的种类
"""

# 常用科学计算包
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# 鸢尾花数据库
from sklearn.datasets import load_iris
# sklearn中的训练集与数据集分离包
from sklearn.model_selection import train_test_split
# 近邻算法包
from sklearn.neighbors import KNeighborsClassifier

# 将鸢尾花数据实例化
iris_dataset = load_iris()

# 分出训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
# 采用k近邻算法构建模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 评估模型
# score方法
knn.score(X_test, y_test)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
# 自编方法
y_predict = knn.predict(X_test)
np.mean(y_predict == y_test)
print("Test set score: {:.2f}".format(np.mean(y_predict == y_test)))

# 数据预测
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
"""
2019.7.7
机器学习基本步骤：
    将数据分出训练集与测试集、选定算法构建模型、用测试集评估模型、分类边界与数据可视化、对未知数据进行预测
学习总结：
    近邻算法本质为对要预测的数据点周围的类别进行统计，取其种类最多的标签作为该预测点的标签值
    numpy数组的练习、函数format的使用方法、代码的单行调试方法、调试数据的查看
存在疑问：
    train_test_split函数的输入参数格式为数组类型，多个样本（每个样本为一个数组分量），每个样本中包含多个特征
理论总结：
    监督学习中通用的函数：fit()函数、类实例化
"""