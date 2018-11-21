import numpy as np

"""买苹果 --------> 买香蕉"""
dataset_filename = "../data/affinity_dataset.txt"
data = np.loadtxt(dataset_filename)
# 文件affinity_dataset.txt是生成的数据，得我们来指定列
# 购买苹果的数量
num_apple_buy = 0
# 符合既买苹果又买香蕉的
rule_valid = 0
# 买苹果不买香蕉的
rule_invalid = 0

for sample in data:
    if sample[3] == 1:
        num_apple_buy += 1
        if sample[4] == 1:
            rule_valid += 1
        else:
            rule_invalid += 1

print("买苹果的有{0}人".format(num_apple_buy))
print("买苹果的又买香蕉有{0}人".format(rule_valid))
print("买苹果的不买香蕉有{0}人".format(rule_invalid))
print("买苹果又买香蕉的支持度{0}".format(rule_invalid / (rule_valid + rule_invalid)))
print("买苹果又买香蕉的置信度为{0}".format(rule_valid / num_apple_buy))