"""
多层感知机（multilayer perceptron, MLP）模型的建立:又可叫做普通（前置）神经网络、深度学习
"""

# 常用科学计算包
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import load_breast_cancer
# sklearn中的训练集与数据集分离包
from sklearn.model_selection import train_test_split
# 多层感知机算法包
from sklearn.neural_network import MLPClassifier

# 数据实例化
cancer = load_breast_cancer()

# 测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# 对数据进行缩放
# 缩放后训练集的mean = 0，std = 1
# 计算训练集中每个特征的平均值
mean_on_train = X_train.mean(axis=0)
# 计算训练集中每个特征的标准差
std_on_train = X_train.std(axis=0)
# 将训练集特征减去平均值，然后乘以标准差的倒数
X_train_scaled = (X_train - mean_on_train) / std_on_train
# 对测试集进行相同的变换（采用的mean和std与训练集相同）
X_test_scaled = (X_test - mean_on_train) / std_on_train

# 多层感知机模型建立
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 100), activation="relu", solver='adam', alpha=0.01)
mlp.fit(X_train_scaled, y_train)

# 模型评估
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# 数据可视化
# 第一隐层权重热图
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

"""
2019.7.12+13
学习总结：
    1.采用神经网络构建模型对乳腺癌判定进行预测，有两个重要参数：hidden_layer_sizes（隐单元层数与节点个数）、alpha（正则化参数）
    该算法对数据要求较高，需要进行数据预处理，使其分布较为紧凑：mean = 0，std = 1，数据之间不能相差太大（与支持向量机类似）
    2.复习matplotlib.pyplot类的功能：
    # 定义画图窗口大小
    plt.figure(figuresize=(20, 5))
    # 对数据矩阵进行热图可视化（全程：image show）
    plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
    # 定义纵坐标标签
    plt.yticks(range(30), cancer.feature_names)
    # 定义坐标轴标签
    plt.xlabel("Columns in weight matrix")
    plt.ylabel("Input feature")
    # 显示深度条
    plt.colorbar()
    3.熟悉机器学习的基本流程，牢记python中的数组是一行一行的，数组并不是向量，要与DataFrame中的一列区分开
需要回忆：
    对书中出现的数据可视化代码进行查找，并进一步改写此代码
    有必要对最近集中数据可视化方式进行一个详细的总结
理论总结：
    多层感知机可视为广义的线性模型，执行多层处理后得到结论，只是对每个隐单元结果都应用了一个非线性函数
    大部分深度学习算法都经过精确调整，只适用于特定的使用场景（因为对于特定的场景进行了特定的参数调节）
"""