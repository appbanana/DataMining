import operator
import numpy as np
from collections import defaultdict

# 训练样本
sample = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
# 训练样本所属分类
label = ['A', 'A', 'B', 'B']


def classify(data, sample, label, k=1):
    print('sample' * 10)
    print(sample.shape)
    print(sample)
    print('sample' * 10)
    print('\n')

    # 训练样本集的行数
    sample_size = sample.shape[0]
    # 将data扩展到和训练样本集sample一样的行数
    data_mat = np.tile(data, (sample_size, 1))
    delta = (data_mat - sample) ** 2
    distance = np.sum(delta, axis=1) ** 0.5
    sorted_distance = np.argsort(distance)

    class_count = defaultdict(int)
    for i in range(k):
        voted_label = label[sorted_distance[i]]
        class_count[voted_label] += 1
    result = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return result[0][0]


if __name__ == '__main__':
    ret = classify([10, 0], sample, label, 3)
    print(ret)
