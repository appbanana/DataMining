from collections import defaultdict
import operator
import numpy as np


def file_2_matrix(filename):
    """
    :param filename: 文件路径
    :return: 返回训练样本和分类数据
    """
    lines = np.loadtxt(filename)
    sample = lines[:, :3].astype('float')
    label = lines[:, -1]
    return sample, label


def classify(data, sample, label, k):
    """
    :param data: 测样本数据
    :param sample: 训练样本数据
    :param label: 训练样本中分类的数据
    :param k: 选取与测试样本距离最近的前k个训练样本
    :return: 测样本所属的分类
    """
    sample_size = sample.shape[0]
    data = np.tile(data, (sample_size, 1))
    delta = (data - sample) ** 2
    distance = np.sum(delta, axis=1) ** 0.5
    sort_distance = np.argsort(distance)
    class_count = defaultdict(int)
    for i in range(k):
        vote_label = label[sort_distance[i]]
        class_count[vote_label] += 1
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sort_class_count[0][0]


def normalize(data):
    # 等价于np.amin(data, axis=0) 每一列中获取最小的值
    min_value = data.min(0)
    # 等价于np.amax(data, axis=0) 每一列中获取最大的值
    max_value = data.max(0)
    value_range = max_value - min_value
    norm_data = np.zeros(np.shape(data))
    data_size = np.shape(data)[0]
    # 也不知道这货到底要干什么
    norm_data = data - np.tile(min_value, (data_size, 1))
    norm_data = norm_data / np.tile(value_range, (data_size, 1))
    return norm_data, value_range, min_value


if __name__ == '__main__':
    # datingTestSet2由自己手动输入 只是为了演示 重在思路
    # 获取训练样本和所属分类
    sample_data, class_label = file_2_matrix('../data/datingTestSet2.txt')
    print(len(sample_data))
    norm_sample_data, value_range_data, min_Value_data = normalize(sample_data)
    k = norm_sample_data.shape[0]
    num = int(k * 0.3)
    error_count = 0.0
    for i in range(num):
        result = classify(norm_sample_data[i, :], norm_sample_data[num:, :], class_label[num:], 6)
        print("The classifier came back with: %d, the real answer is %d" % (result, class_label[i]))

        if result != class_label[i]:
            error_count += 1
    print("The total error rate is %f " % (error_count / float(num)))
