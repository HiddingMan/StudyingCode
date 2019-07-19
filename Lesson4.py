"""
决策树集成模型的建立：随机森林、梯度提升回归树（梯度提升机）
"""

# 常用科学计算包
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import load_breast_cancer
# sklearn中的训练集与数据集分离包
from sklearn.model_selection import train_test_split
# 随机森林算法包
from sklearn.ensemble import RandomForestClassifier
# 梯度提升回归树算法包
from sklearn.ensemble import GradientBoostingClassifier

# 数据实例化
cancer = load_breast_cancer()

# 测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                    random_state=0)

# 随机森林模型建立
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
# 模型评估
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

# 梯度提升回归树模型建立
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.1)
gbrt.fit(X_train, y_train)
# 模型评估
print("Accuracy on training set0: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set0: {:.3f}".format(gbrt.score(X_test, y_test)))


"""
2019.7.10
学习总结：
    采用决策树集成模型对乳腺癌判定进行预测：随机森林、梯度提升回归树
    由于单一决策树容易发生过拟合现象，可以采用决策树集成的方法，降低原来的过拟合现象
    随机森林算法中决策树数量越多越好，梯度提升机可以限制决策树的最大深度和学习强度来降低其过拟合现象（该算法
    对参数较为敏感）
    熟悉机器学习的基本流程
需要回忆：
    对书中出现的数据可视化代码进行查找，并进一步改写此代码
    python基本数据类型基础：列表（list）、数组（numpy.array）、矩阵（DateFrame）
"""