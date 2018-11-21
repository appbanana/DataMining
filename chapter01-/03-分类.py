import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
"""
参考链接： https://blog.csdn.net/red_stone1/article/details/80607960
我们知道,一般机器学习算法包括两个过程：训练过程和测试过程。训练过程通过使用机器学习算法在训练样本上迭代训练，
得到较好的机器学习模型；测试过程是使用测试数据来验证模型的好坏，通过正确率来呈现。
kNN算法的本质是在训练过程中，它将所有训练样本的输入和输出标签(label)都存储起来。
测试过程中，计算测试样本与每个训练样本的距离，选取与测试样本距离最近的前k个训练样本。
然后对着k个训练样本的label进行投票，票数最多的那一类别即为测试样本所归类。
--------------------- 
"""
# 分类 聚类 回归
data_filename = "../data/ionosphere.data"
X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

with open(data_filename, 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        # 去除最后一位 因为最后一位为'g' 或'b'，是无效数据
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        y[i] = row[-1] == 'g'

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

print("训练集数据有 {} 条".format(X_train.shape[0]))
print("测试集数据有 {} 条".format(X_test.shape[0]))
print("每条数据有 {} 个features".format(X_train.shape[1]))

# 实例化算法对象->训练->预测->评价
estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)
y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100

print("准确率{0:.1f}%".format(accuracy))

# 其他评价方式
scores = cross_val_score(estimator, X, y, scoring='accuracy', cv=5)
average_accuracy = np.mean(scores) * 100
print("平均准确率{0:0.1f}%".format(average_accuracy))

avg_scores = []
all_scores = []
paramrter_values = list(range(1, 21))
for n_neighbors in paramrter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy', cv=5)
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

plt.figure(figsize=(32, 20))
plt.plot(paramrter_values, avg_scores, '-o', linewidth=5, markersize=24)