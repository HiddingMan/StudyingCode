"""
k均值聚类算法（k-means算法）
"""
# 常用科学计算包
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import make_moons
# 导入k-means算法包
from sklearn.cluster import KMeans
# 导入PCA算法包
# from sklearn.decomposition import PCA
# 导入NMF算法包
# from sklearn.decomposition import NMF

# 数据实例化
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# k-means分解
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# 数据k-means分类可视化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60, marker='^', c=range(kmeans.n_clusters),
            linewidths=2, cmap='Paired')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
print("Cluster memberships:\n{}".format(y_pred))
"""
2019.7.21
k-means聚类算法学习总结：
    1.聚类算法的目标是划分数据，与分类算法目标类似
    2.kmeans算法对于一些数据并不适用，其只适用于紧凑分布的数据，对于长条形数据不是很实用。
    3.矢量量化（看作数据分解的一种方式）：可以将二维数据点分成n个簇(n维)，只有表示该点对应的簇中心的那个特征不为零，
    即可形成多特征数据。相当于实现了数据的特征增加，之后可以对新特征数据进行线性模型的建立，分类划分两个半月形数据。
    4.kmeans算法中要指定簇的个数，在现实情况中我们是不知道簇的个数的。
数据可视化复习：
    # scatter散点图中可用c进行不同的颜色分类
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60, marker='^', 
    c=range(kmeans.n_clusters),linewidths=2, cmap='Paired')
"""