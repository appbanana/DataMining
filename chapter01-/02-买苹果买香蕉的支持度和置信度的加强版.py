import numpy as np
from collections import defaultdict
from operator import itemgetter

"""买苹果 --------> 买香蕉"""
dataset_filename = "../data/affinity_dataset.txt"
data = np.loadtxt(dataset_filename)
# 文件affinity_dataset.txt是生成的数据，得我们来指定列
features = ["bread", "milk", "cheese", "apples", "bananas"]

# 符合既买xx又买yy
valid_rules = defaultdict(int)
# 买xx不买yy的
invalid_rules = defaultdict(int)
# 买xx
num_occurences = defaultdict(int)

for sample in data:
    for i in range(len(features)):
        if sample[i] == 0:
            continue
        # 买xx的计作规则X
        num_occurences[i] += 1
        for j in range(len(features)):
            if i == j:
                continue
            if sample[j] == 1:
                # 买xx 又买yy
                valid_rules[(i, j)] += 1
            else:
                invalid_rules[(i, j)] += 1

support = valid_rules
confidence = defaultdict(float)
for i, j in valid_rules.keys():
    confidence[(i, j)] = valid_rules[(i, j)] / num_occurences[i]


# for i, j in confidence.keys():
#     food = features[i]
#     another_food = features[j]
#     print("Rule:买{0}又买{1}".format(food, another_food))
#     print("-支持度Support:{0:.3f}".format(valid_rules[(i, j)]))
#     print("-自信度度Confidence:{0:.3f}".format(confidence[(i, j)]))
#     print("\n")

def print_rule(index, another_index, features, confidence, valid_rules):
    food = features[index]
    another_food = features[another_index]
    print("Rule:买{0}又买{1}".format(food, another_food))
    print("-支持度Support:{0:.3f}".format(valid_rules[(index, another_index)]))
    print("-自信度度Confidence:{0:.3f}".format(confidence[(index, another_index)]))


# print_rule(1, 3, features, confidence, valid_rules, invalid_rules)
sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print_rule(premise, conclusion, features, confidence, support)
