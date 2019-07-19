"""
决策树模型的建立
"""

# 常用科学计算包
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带乳腺癌数据库
from sklearn.datasets import load_breast_cancer
# sklearn中的训练集与数据集分离包
from sklearn.model_selection import train_test_split
# 决策树算法包
from sklearn.tree import DecisionTreeClassifier
# 决策树可视化包
from sklearn.tree import export_graphviz

# 数据实例化
cancer = load_breast_cancer()

# 测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                    random_state=42)
# 决策树模型建立
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

# 模型评估
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# 导出决策树文件
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names,impurity=False, filled=True)

# 导入决策树文件
# import graphviz
# with open("tree.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)

# 决策树的特征重要性
print("Feature importances:\n{}".format(tree.feature_importances_))


# 定义函数用于特征重要性的可视化
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


# 特征重要性可视化
plot_feature_importances_cancer(tree)
"""
2019.7.9
学习总结：
    采用决策树模型对乳腺癌判定进行预测，即什么特征能够判定乳腺癌（相关性更强）
    单一决策树的缺点即为容易过拟合，可以采用预剪枝进行限制，提高模型的泛化能力
    熟悉机器学习的基本流程
需要回忆：
    条形图的绘制
    range（4）函数，return多个区间：[0-1][1-2][2-3][3-4]
    numpy.arange(4)函数，return多个数：0，1，2，3
    修改坐标节点标签：plt.yticks(放置位置, 放置内容)
    python基本数据类型基础：列表（list）、数组（numpy.array）、矩阵（DateFrame）
"""