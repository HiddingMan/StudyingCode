"""
DBSCAN聚类算法（density-based spatial clusting of applications with noise 算法：具有噪声的基于密度的空间聚类应用算法）
"""
# 常用科学计算包
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import make_blobs, make_moons
# 导入DBSAN算法包
from sklearn.cluster import DBSCAN
# 导入PCA算法包
# from sklearn.decomposition import PCA
# 导入NMF算法包
# from sklearn.decomposition import NMF
# 导入数据预处理包
from sklearn.preprocessing import StandardScaler

# 数据实例化
X, y = make_blobs(random_state=0, n_samples=12)

# 算法应用
dbsan = DBSCAN()
clusters = dbsan.fit_predict(X)
print("cluster memberships:\n{}".format(clusters))      # 结果-1为噪声(调节合适的min_samples和eps能够使结果更好，或者进行数据预处理)

# 数据预处理
X0, y0 = make_moons(n_samples=200, noise=0.05, random_state=0)
# 对数据进行standardscaler缩放
scaler = StandardScaler()
scaler.fit(X0)
X_scaled = scaler.transform(X0)

dbsan0 = DBSCAN()
clusters0 = dbsan0.fit_predict(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters0, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

"""
2019.7.23
DBSAN聚类算法学习总结：
    1.DBSAN聚类与凝聚聚类算法类似，同样无法对新数据进行预测，即没有predict函数，只有fit函数，但也具有fit_predict函数
    2.DBSAN算法与k-means、凝聚聚类算法不同在于：不需要人工给定簇的个数
    3.两个重要参数：最小样本数（n_samples= ）、查询半径（eps= ），默认值分别为
复习数据预处理standardscaler方法
"""