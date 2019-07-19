"""
多分类线性模型的建立
"""

# 常用科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 画图工具包seaborn
import seaborn as sns
# scikit-learn自带练习数据库
from sklearn.datasets import make_blobs
# 回归算法包
from sklearn.linear_model import LogisticRegression
# 支持向量机算法包
from sklearn.svm import LinearSVC

# 数据实例化
X, y = make_blobs(random_state=42)
# 将数据类型转换为DateFrame，便于数据可视化
df = pd.DataFrame(X, columns=['a', 'b'])
df0 = pd.DataFrame(y)
df['c'] = df0

# 模型建立
# 采用回归算法构建模型
knn = LogisticRegression().fit(X, y)
# 散点分类显示
sns.FacetGrid(df, hue='c', size=6).map(plt.scatter, 'a', 'b').add_legend()
# 画出线性模型边界
line = np.linspace(-15, 15)
for coef, intercept, color in zip(knn.coef_, knn.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)

# 采用线性支持向量机算法构建模型
knn_svc = LinearSVC().fit(X, y)
# 散点分类显示
sns.FacetGrid(df, hue='c', size=6).map(plt.scatter, 'a', 'b').add_legend()
# 画出线性模型边界
line = np.linspace(-15, 15)
for coef, intercept, color in zip(knn_svc.coef_, knn_svc.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.show()
"""
2019.7.8
学习总结：
    采用线性模型（线性支持向量机、回归算法）对多种类进行预测，并划分边界
    回忆将散点图分类表示，简单回忆seaborn函数库
    对于for循环中存在多个参数，可使用zip()函数来进行多参数对应赋值
    熟悉机器学习的基本流程
需要回忆：
    数据可视化方法
    python基本数据类型基础
理论总结：
    线性模型的分类问题本质上很简单，就是一个输入特征的函数（决策边界）：相当于判断一个函数（决策边界）大于还是小于零的问题，
    多分类问题就是二分类问题的扩展（相当于进行了排列组合）！
    线性分类的回归问题本质上更简单，就是一个输入特征的函数 y ！
常见的监督学习模型：
    线性模型是支持向量机、神经网络的本质基础！
    决策树模型是决策树集成模型（随机森林、梯度提升回归树）的本质和基础！
    还有一种与线性模型较为相似的模型：朴素贝叶斯模型！
"""
