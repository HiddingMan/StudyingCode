"""
流形学习算法（manifold learning algorithm）中的t-SNE算法（主要用于数据可视化，用于数据探索）并对比PCA降维
"""
# 常用科学计算包
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import load_digits
# 导入PCA算法包
from sklearn.decomposition import PCA
# 导入t-SNE算法包
from sklearn.manifold import TSNE
# 数据实例化
digits = load_digits()

# 初始整体图像数据可视化
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)

# PCA降维
pca = PCA(n_components=2)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)
# 降维后投影数据可视化
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 1].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color=colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

# t-SNE降维
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)
# 降维后数据可视化
plt.figure()
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 1].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), color=colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")
plt.show()

"""
2019.7.20
t-SNE算法学习总结：
    1.流形学习算法能够使原始空间中距离较近的点更加靠近，远离的点更加远离，换句话说，该方法试图保存那些表示哪些点\
    比较靠近的信息。相比于PCA降维算法来说，流形学习方法能够更好的进行分类。PCA方法是找到整体数据集的主方向，再将所有数据点进行\
    主方向投影，相似的图像在主方向投影的坐标表示也是相似的，但流形算法更能够突出数据集自身的多特征相似性，使相近的数据集更加紧凑\
    能够更好的进行分类
    2.流形学习的缺点：该方法不能进行对测试集进行变换（不允许变换新数据）
"""