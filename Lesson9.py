"""
主成分分析提取数字图像特征脸
"""
# 常用科学计算包
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# scikit-learn自带数据库
from sklearn.datasets import fetch_lfw_people
# sklearn中的训练集与数据集分离包
from sklearn.model_selection import train_test_split
# 导入近邻算法包
from sklearn.neighbors import KNeighborsClassifier
# 导入pca算法包
from sklearn.decomposition import PCA

# 数据实例化
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

# 图像可视化
fix, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

# 输出图像数据信息
print("people.images.shape: {}".format(people.images.shape))
print("number of class: {}".format(len(people.target_names)))

# 计算每个人物出现的次数
counts = np.bincount(people.target)
# 将次数与名称一起输出
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='     ')
    if (i + 1) % 3 == 0:
        print()

# 为了降低数据偏斜，每人最多50张图像
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
# 数据缩放到0-1之间
X_people = X_people / 255

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

# 对训练集进行PCA分解
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit(X_train)
# 将数据集投影到主成分方向
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 采用近邻算法进行模型建立
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
# 模型评估
print("模型准确度:{}".format(knn.score(X_test_pca, y_test)))

# 模态可视化
fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), camp='viridis')
    ax.set_title("{}.component".format((i + 1)))
"""
2019.7.18
学习总结：
    1.此案例首先选出合理的测试集数据，其次将测试集数据进行pca分解得到主成分，再将训练集和测试集数据向主成分\
    投影，得到新的数据，最后使用该数据进行近邻算法分类模型的建立

学习python新语法：enumerate、新判断语句
1:  for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='     ')
    if (i + 1) % 3 == 0:
        print()

2:  for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), camp='viridis')
    ax.set_title("{}.component".format((i + 1)))
    
3：  mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]

统一总结：
    找主要特征可以采用PCA主成分投影的方法
"""



