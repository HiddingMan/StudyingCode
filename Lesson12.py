"""
凝聚聚类算法（agglomerative clustering算法）
"""
# 常用科学计算包
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import make_blobs
# 导入agglomerative clustering算法包
from sklearn.cluster import AgglomerativeClustering
# 导入PCA算法包
# from sklearn.decomposition import PCA
# 导入NMF算法包
# from sklearn.decomposition import NMF
# 导入SciPy科学计算包
from scipy.cluster.hierarchy import dendrogram, ward

# 数据实例化
X, y = make_blobs(random_state=1)

# 算法应用
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

# 数据分类可视化
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# 层次聚类与树状图
X0, y0 = make_blobs(random_state=0, n_samples=12)
# 将ward聚类应用于数组X0，scipy的ward函数返回一个数组，指定执行凝聚聚类时跨越的距离
linkage_array = ward(X0)
# 现在为包含簇之间距离的linkage_array绘制树状图
plt.figure()
dendrogram(linkage_array)
# 在树中标记划分成两个簇或者三个簇的位置
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, 'two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, 'three clusters', va='center', fontdict={'size': 15})
plt.xlabel("sample index")
plt.ylabel("cluster distance")
plt.show()
"""
2019.7.22
agglomerative clustering 聚类算法学习总结：
    1.凝聚聚类无法对新数据进行预测，即没有predict函数，只有fit函数，但也具有fit_predict函数
    2.agglomerative clustering 算法与k-means算法类似，也要指定簇的个数，在现实情况中我们是不知道簇的个数的。
    3.相比于k-means聚类，凝聚聚类能够更好的看出层次聚类和树状图，整体直观性更强，但k-means聚类能够对新数据集进行预测
新学习数据可视化方法：
    # 在树中标记划分成两个簇或者三个簇的位置
    ax = plt.gca()
    # 获得边界
    bounds = ax.get_xbound()
    # 在边界贯穿划线
    ax.plot(bounds, [7.25, 7.25], '--', c='k')
    ax.plot(bounds, [4, 4], '--', c='k')
    # 对边界固定位置添加文本
    ax.text(bounds[1], 7.25, 'two clusters', va='center', fontdict={'size': 15})
    ax.text(bounds[1], 4, 'three clusters', va='center', fontdict={'size': 15})
"""