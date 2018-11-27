#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:38:33 2017
电影推荐分析:
    使用 亲和性分析方法 基于 Apriori算法 推荐电影
@author: yingzhang
"""

# 读取数据集: http://grouplens.org/datasets/movielens/
import os
# 使用pandas加载数据
import pandas as pd

'''
1m数据集读取方法
'''
# data_folder=os.path.join( os.path.expanduser("~"),"ml-1m")
# ratings_filename=os.path.join( data_folder,"ratings.dat")
# all_ratings=pd.read_csv( ratings_filename, delimiter="::",header=None, names=["UserID","MovieID","Rating","Datetime"])
'''
100k数据集读取方法
'''
data_folder = os.path.join(os.path.expanduser("~"), "ml-100k")
# 获取用户评分数据
ratings_filename = os.path.join(data_folder, "u.data")
all_ratings = pd.read_csv(ratings_filename, delimiter="\t", header=None,
                          names=["UserID", "MovieID", "Rating", "Datetime"])

all_ratings[:1]
# 输出的数据格式如下
'''
   UserID  MovieID  Rating   Datetime
0       1     1193       5  978300760
'''
# 时间格式要转换一下
all_ratings["Datetime"] = pd.to_datetime(all_ratings["Datetime"], unit='s')
all_ratings[:1]

# 新增一列，用来存用户对某个电影是否喜欢 ( 如果评分大于3)
all_ratings["Favorable"] = all_ratings["Rating"] > 3
all_ratings[:10]
# 输出的数据格式如下: Favorable这一列的数据表明用户是否喜欢这部电影
'''
   UserID  MovieID  Rating            Datetime Favorable
0       1     1193       5 2000-12-31 22:12:40      True
1       1      661       3 2000-12-31 22:35:09     False
'''
# 从数据集中取前200名用户的打分数据作训练集
ratings = all_ratings[all_ratings['UserID'].isin(range(200))]
# 过滤一次数据，只保留用户喜欢的电影(即 Favorable为True值的)
favorable_ratings = ratings[ratings["Favorable"]]
favorable_ratings[:5]

# 因为要生成频繁项集，所以我们只对打分超过一次的用户感兴趣，所以按照UserID分组，并遍历每个用户看过的每一部电影，存到一个字典中
from collections import defaultdict

favorable_reviews_by_users = dict((k, frozenset(v.values))
                                  for k, v in favorable_ratings.groupby("UserID")["MovieID"])
print("length: {0}".format(len(favorable_reviews_by_users)))

# 再创建一个数据框，存入每部电影评价分为3分以上的人数( 即 Favorable为True)的数量
num_favorable_by_movie = ratings[["MovieID", "Favorable"]].groupby("MovieID").sum()
# 查看结果
num_favorable_by_movie
# 排序输出前五名
num_favorable_by_movie.sort("Favorable", ascending=False)[:5]

# 创建一个函数，它接收新发现的频繁项集，创建超集，检测频繁程度
'''
favorable_reviews_by_users: 用户打分情况的集合
k_1_itemsets: 上一个频繁项集
min_support:最小支持度
返回值格式:
    dict( 频繁项集    支持度 )
'''


def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    # 循环用户和他们的打分数据
    for user, reviews in favorable_reviews_by_users.items():
        # 再循环前一次找出的频繁项集,判断它们是否为 reviews的子集,
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):  # 如果是，表明用户已经为子集中的电影打过分了
                # 那么接下来，就要遍历用户打过分却没有出现在项集reviews中的电影了,因为这样可以用它来生成超集，更新该项集的计数
                for other_reviewed_movie in reviews - itemset:  # other_reviewed_movie 用户打过分，但还不在频繁项集中
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1  # 这个频繁项集的数量支持度+1
    # 函数最后检测达到支持度要求的项集，只返回频繁度够的频繁项集
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])


import sys

# 创建一个字典，存不同长度的频繁项集
# 数据格式:
# 频繁项集长度   对应的频繁项集
frequent_itemsets = {}
min_support = 50  # 要求的最小支持度
# 从频繁项集长度为1的开始,并且支持度要大于50
frequent_itemsets[1] = dict(
    (frozenset((movie_id,)), row["Favorable"]) for movie_id, row in num_favorable_by_movie.iterrows() if
    row["Favorable"] > min_support)
# 输出频繁集长度为1,支持度大于50的所有的电影信息
frequent_itemsets[1]
print("there are {0} movie with more than {1} favorable reviews".format(len(frequent_itemsets[1]), min_support))
sys.stdout.flush()  # 将缓冲区的内容输出到终端

# 定义要找的频繁集的最大长度
max_length = 20
# 循环频繁集长度从2到 max_length
for k in range(2, max_length):
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k - 1], min_support)
    if len(cur_frequent_itemsets) == 0:
        print("can not find any frequent itemsets of length {0}".format(k))
        sys.stdout.flush()
        break
    else:
        print(" find {0} frequent itemsets of length {1}".format(len(cur_frequent_itemsets), k))
        print("\t data as following:")
        # print( cur_frequent_itemsets )
        sys.stdout.flush()
        frequent_itemsets[k] = cur_frequent_itemsets
# del itemsets of length 1
# del frequent_itemsets[1]

#######################################################################
# 以上Apriori算法结束后，得到了一系列的频繁项集，但它还不是关联规则。频繁项集是一组达到最小支持度的项目，而关联规则是由前提和结论组成
# 从频繁项集中抽取关联规则，把其中几部电影作为前提，另一部电影作为结论组成规则: 如果用户喜欢xx,yy,zz,那么他们也会喜欢ttt
# 遍历频繁项集，为每个项集生成规则
candidate_rules = []
# itemset_length 频繁项集长度
# itemset_counts :  (itemset,frequency)
for itemset_length, itemset_counts in frequent_itemsets.items():
    # 取出itemset_counts中的每个键，{电影1,电影2,...}
    for itemset in itemset_counts.keys():
        # 循环频繁项集中的每部电影，生成条件和结论
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print("there are {0} candidate rules".format(len(candidate_rules)))
# print("the rules as following:")
# print( candidate_rules)

#######################################################################
# 计算每条规则的置信度:
# 先用两个字典存规则应验, 规则不适用数量
correct_counts = defaultdict(int)  # 规则应验
incorrect_counts = defaultdict(int)  # 规则不适用
# 循环所有的用户及他们喜欢的电影
for user, reviews in favorable_reviews_by_users.items():
    # 循环所有的规则
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        # 判断前提是否是 reviews中的一个子集， 并且结论也在 reviews中，说明这条规则应验，否则不适用
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
# 计算置信度
rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(
    correct_counts[candidate_rule] + incorrect_counts[candidate_rule]) for candidate_rule in candidate_rules}
# 设定最低置信度
min_confidence = 0.9
# 过滤掉小于最低置信度的规则
rule_confidence = {candidate_rule: confidence for candidate_rule, confidence in rule_confidence.items() if
                   confidence > min_confidence}
print("the total of rules which bigger than min_confidence is {}".format(len(rule_confidence)))

# 排序输出前五条置信度最高的规则
from operator import itemgetter

sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print("Rule: if a person recommends {0} they will also recommend {1}".format(premise, conclusion))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")

#######################################################################
# 加载电影的名字
# 100k数据集取法
movie_name_filename = os.path.join(data_folder, "u.item")
movie_name_data = pd.read_csv(movie_name_filename, delimiter="|", header=None, encoding="mac-roman")
movie_name_data.columns = ["MovieID", "Title", "Release Date", "Video Release", "IMDB", "<UNK>", "Action", "Adventure",
                           "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                           "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


# 定义一个查找电影名的函数
def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data["MovieID"] == movie_id]["Title"]
    title = title_object.values[0]
    return title


# 测试这个函数
get_movie_name(4)

# 重新排序输出前五条置信度最高的规则
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ",".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: if a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")

#######################################################################
# 评估: 寻找最好的规则.
# 抽取所有没有用于训练的数据作为测试集, 训练集数据用了前200名用户的打分数据，测试集用其它的数据即可
test_dataset = all_ratings[~all_ratings['UserID'].isin(range(200))]
test_favorable_ratings = test_dataset[test_dataset["Favorable"]]
test_favorable_reviews_by_users = dict((k, frozenset(v.values))
                                       for k, v in test_favorable_ratings.groupby("UserID")["MovieID"])
# 计算规则应验的数量
test_correct_counts = defaultdict(int)  # 规则应验
test_incorrect_counts = defaultdict(int)  # 规则不适用
# 循环所有的用户及他们喜欢的电影
for user, reviews in test_favorable_reviews_by_users.items():
    # 循环所有的规则
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        # 判断前提是否是 reviews中的一个子集， 并且结论也在 reviews中，说明这条规则应验，否则不适用
        if premise.issubset(reviews):
            if conclusion in reviews:
                test_correct_counts[candidate_rule] += 1
            else:
                test_incorrect_counts[candidate_rule] += 1

            # 计算置信度
test_rule_confidence = {candidate_rule: test_correct_counts[candidate_rule] / float(
    test_correct_counts[candidate_rule] + test_incorrect_counts[candidate_rule]) for candidate_rule in candidate_rules}
print(len(test_rule_confidence))
# 最后排序输出前五名
sorted_test_confidence = sorted(test_rule_confidence.items(), key=itemgetter(1), reverse=True)
print(sorted_test_confidence[:5])

# 输出规则信息
for index in range(10):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ",".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: if a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Train Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print(" - Test Confidence: {0:.3f}".format(test_rule_confidence[(premise, conclusion)]))
    print("")
