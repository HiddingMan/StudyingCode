"""
主成分分析（PCA：principal component analysis）
"""

# 常用科学计算包
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# scikit-learn自带数据库
from sklearn.datasets import load_breast_cancer
# 导入数据预处理算法包
from sklearn.preprocessing import StandardScaler
# 导入pca算法包
from sklearn.decomposition import PCA

# 数据实例化
cancer = load_breast_cancer()

# 使用standardscaler算法对数据进行缩放
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

# 对目标数据进行PCA降维，保留前两个主成分
pca = PCA(n_components=2)
pca.fit(X_scaled)
# 将数据投影到保留的主成分（pod中的单阶模态重构）
X_pca = pca.transform(X_scaled)

# 输出降维之前与降维之后的单阶重构数据
print("原始数据形状:{}".format(str(X_scaled.shape)))
print("转换后结果:{}".format(str(X_pca.shape)))

# 采用第一主成分和第二主成分作图
X_pca0 = pd.DataFrame(X_pca, columns=['a', 'b'])
X_pca0['c'] = cancer.target
# 散点分类显示
sns.FacetGrid(X_pca0, hue='c', size=6).map(plt.scatter, 'a', 'b')
plt.legend(["Bad", "Good"], loc='best')
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

# pac系数可视化(模态展示)
plt.figure()
mm = pca.components_
plt.matshow(pca.components_, cmap="viridis")
plt.show()
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal component")
plt.show()
"""
2019.7.17
学习总结：
    1.pca降维对应于流体中常用的pod分解方法，本质相同，与数据预处理类似，具有两个常用函数：fit()、transform()
    2.pca降维的缺点：由于是对原始数据整体进行分解，所以不能很好的对前两个主成分（坐标轴）进行解释（因为是整体的特征分解结果）
    3.pca分解的模态系数保存在pca.components_中

回忆数据可视化：对二维数据进行分类（与Lesson2同）
1:  # 设置列标签
    X_pca0 = pd.DataFrame(X_pca, columns=['a', 'b'])
    X_pca0['c'] = cancer.target
    # 散点分类显示
    sns.FacetGrid(X_pca0, hue='c', size=6).map(plt.scatter, 'a', 'b')
    plt.legend(["Bad", "Good"], loc='best')
    plt.gca().set_aspect("equal")
    
2:  # 热图新函数：matshow（与imshow方法类似）
    plt.matshow(pca.components_, cmap="viridis")
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
    
统一总结：
    降维的目的之一就是实现数据二维可视化
"""

