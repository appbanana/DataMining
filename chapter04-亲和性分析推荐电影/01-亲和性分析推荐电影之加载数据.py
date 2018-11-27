from datetime import datetime
import operator
import sys
# import itertools
import pandas as pd
from collections import defaultdict


def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    # 遍历所有用户和打过分的电影
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            # 如果一个集合的子集不是频繁项集 那么这个子集的超集一定不是频繁项集
            if itemset.issubset(reviews):
                # 用户打过分 但没有出现在itemset子集中 然后在itemset和剩下的元素进行组合
                for other_review_moive in (reviews - itemset):
                    current_superset = itemset | frozenset((other_review_moive,))
                    counts[current_superset] += 1
    return dict([(itemset, frequent) for itemset, frequent in counts.items() if frequent >= min_support])


def get_moive_name(moive_id, moive_data):
    moive_obj = moive_data[moive_data['MovieID'] == moive_id]
    return moive_obj['Title'].values[0]


if __name__ == '__main__':
    start = datetime.now()
    file_path = '../data/u.data'
    # usecols 读取那几列数据
    # names 为数据添加列名
    # nrows=1000000 读取数据多少条数据 我是取了100万条数据 当你你可以不设置这个
    # memory_map 如果file_path是文件路径的话，直接把文件内容映射到内存中 直接从内存中读取数据，提高性能
    all_ratings = pd.read_csv(file_path, delimiter='\t',
                              header=None,
                              names=["UserID", "MovieID", "Rating", "Datetime"],
                              memory_map=True)
    all_ratings['Datetime'] = pd.to_datetime(all_ratings['Datetime'], unit='s', infer_datetime_format=True)
    end = datetime.now()
    print('读取文件花费时间 %s' % (end - start))
    # print(all_ratings.head())
    """
           UserID  MovieID  Rating            Datetime Favorable
    0       1     1193       5 2000-12-31 22:12:40      True
    1       1      661       3 2000-12-31 22:35:09     False
    """

    # 利用Apiori算法来实现
    all_ratings['Favorable'] = all_ratings['Rating'] > 3
    # 选取前200名用户的打分数据作为训练集
    # isin 判断元素在不在数组中 在的话为True 不在的为False
    """
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
    >>> df.isin([1, 3, 12, 'a'])
           A      B
    0   True   True
    1  False  False
    2   True  False
    """

    # 取UserID小于200作为训练集  rating取出来的数据不只200条 因为一个用户可能会评价多部电影  len(rating) = 19531
    rating = all_ratings[all_ratings['UserID'].isin(range(200))]
    favorable_ratings = rating[rating['Favorable']]
    # 把下面的代码转化为 集合表达式 将用户评价的电影进行归类 eg: 1:{66, 99, 100} 1用户评价过66，99，100的电影
    # for key, v in favorable_ratings.groupby('UserID')['MovieID']:
    #     print(key, frozenset(v.values))
    favorable_reviews_by_users = dict(
        (key, frozenset(v.values)) for key, v in favorable_ratings.groupby('UserID')['MovieID'])


    # print('---favorable_reviews_by_users---' * 3)
    # print(favorable_reviews_by_users)
    # print('\n')

    # print('---favorable_reviews_by_users---' * 3)
    # for k, v in rating[['MovieID', 'Favorable']].groupby('MovieID').sum:
    #     print(k, list(v))
    num_favorable_by_movie = rating[['MovieID', 'Favorable']].groupby('MovieID').sum()
    # print(num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5])
    # print('---num_favorable_by_movie---' * 3)
    print(num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5])
    # print('\n')


    # 生成初始的频繁项集 项集大于最小支持度则为频繁项集
    frequent_itemsets = {}
    min_support = 50
    # 将下面代码转化为集合表达式
    # for MovieID, row in num_favorable_by_movie.iterrows():
    #     if row['Favorable'] >= min_support:
    #         print((frozenset((MovieID,)),row["Favorable"]))
    # 一个元素moive_id 满足最小支持度的初始频繁项集
    frequent_itemsets[1] = dict((frozenset((moive_id,)), row['Favorable']) for moive_id, row in
                                num_favorable_by_movie.iterrows() if row['Favorable'] >= min_support)
    # print(frequent_itemsets[2])
    # print('\n')

    # 定义要找的频繁集的最大长度
    max_length = 20
    sys.stdout.flush()
    for k in range(2, max_length):

        cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k - 1],
                                                       min_support)
        if len(cur_frequent_itemsets) == 0:
            print("Did not find any frequent itemsets of length {}".format(k))
            sys.stdout.flush()
            break
        else:
            print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
            # print(cur_frequent_itemsets)
            sys.stdout.flush()
            frequent_itemsets[k] = cur_frequent_itemsets
    # 循环结束后  frequent_itemsets 因为类似下面这种格式
    # frequent_itemsets {1:[(1,), (2,)], 2:[(1,3), (6, 8)], 3:[(1, 6, 8), (1, 8 10)],... ...}
    # We aren't interested in the itemsets of length 1, so remove those
    del frequent_itemsets[1]

    #######################################################################
    # 以上Apriori算法结束后，得到了一系列的频繁项集，但它还不是关联规则。频繁项集是一组达到最小支持度的项目
    # 遍历频繁项集，为每个项集生成规则
    candidate_rules = []
    # key 是频繁项集的长度 value 是频繁项集的集合
    for key, value in frequent_itemsets.items():
        # print((key, value))
        # (2, {frozenset({1, 7}): 62, frozenset({1, 50}): 100, frozenset({56, 1}): 64})
        for itemset in value.keys():
            for conclusion in itemset:
                # 个人认为 这层for 循环有一定的缺陷 这实际上规定了多个对象推断另外一个对象
                premise = itemset - set((conclusion,))
                candidate_rules.append((premise, conclusion))

    # print(candidate_rules)

    #######################################################################
    # 计算每条规则的置信度:
    # 先用两个字典存规则应验, 规则不适用数量
    correct_counts = defaultdict(int)
    incorrect_counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for candidate_rule in candidate_rules:
            premise, conclusion = candidate_rule
            if premise.issubset(reviews):
                if conclusion in reviews:
                    correct_counts[candidate_rule] += 1
                else:
                    incorrect_counts[candidate_rule] += 1
    # 计算置信度
    # rule_confidence = {}
    # for candidate_rule in candidate_rules:
    #     rule_confidence[candidate_rule] = float(correct_counts[candidate_rule]) / (correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
    rule_confidence = {candidate_rule: float(correct_counts[candidate_rule]) / (correct_counts[candidate_rule] + incorrect_counts[candidate_rule]) for candidate_rule in candidate_rules}
    # print(rule_confidence)

    # 设定最低置信度
    mini_confidence = 0.9
    # mini_rule_confidence = {}
    # for key, value in rule_confidence.items():
    #     if value >= mini_confidence:
    #         mini_rule_confidence[key] = value
    mini_rule_confidence = {key: value for key, value in rule_confidence.items() if value >= mini_confidence}
    # print(mini_rule_confidence)

    sorted_confidence = sorted(mini_rule_confidence.items(), key=operator.itemgetter(1), reverse=True)
    for index in range(5):
        print("Rule #{0}".format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        print("Rule: If a person recommends {0} they will also recommend {1} ".format(premise, conclusion))
        print(" - Confidence:{0: .3f}".format(rule_confidence[(premise, conclusion)]))
        print("-------------------")

    #######################################################################
    # 加载电影的名字
    movie_name_filename = '../data/u.item'
    movie_name_data = pd.read_csv(movie_name_filename, delimiter="|", header=None, encoding="mac-roman")
    movie_name_data.columns = ["MovieID", "Title", "Release Date", "Video Release", "IMDB", "<UNK>", "Action",
                               "Adventure",
                               "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                               "Film-Noir",
                               "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    # print(movie_name_data)
    # 测试
    #get_moive_name(4, movie_name_data)
    for index in range(5):
        print("Rule #{0}".format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        premise_names = ",".join(get_moive_name(idx, movie_name_data) for idx in premise)
        conclusion_name = get_moive_name(conclusion, movie_name_data)
        print("Rule: If a person recommends {0} they will also recommend {1} ".format(premise_names, conclusion_name))
        print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
        print("")

    #######################################################################
    # 评估最好的规则
    # 抽取所有没有用于训练集的数据最为测试集，测试正确率
    test_dataset = all_ratings[~all_ratings['UserID'].isin(range(200))]
    test_favorable_ratings = test_dataset[test_dataset['Favorable']]
    test_favorable_reviews_by_users = dict((key, frozenset(value.values)) for key, value in test_favorable_ratings.groupby('UserID')['MovieID'])
    # print('***' * 10)
    # print(test_dataset)
    test_correct_counts = defaultdict(int)
    test_incorrect_counts = defaultdict(int)

    # 循环所有的用户及他们喜欢的电影
    for user, reviews in test_favorable_reviews_by_users.items():
        for candidate_rule in candidate_rules:
            premise, conclusion = candidate_rule
            if premise.issubset(reviews):
                if conclusion in reviews:
                    test_correct_counts[candidate_rule] += 1
                else:
                    test_incorrect_counts[candidate_rule] += 1
    # 计算置信度
    # test_rule_confidence = {candidate_rule: float(test_correct_counts[candidate_rule]) / (test_correct_counts[candidate_rule] + test_incorrect_counts[candidate_rule]) for candidate_rule in candidate_rules}
    test_rule_confidence = {candidate_rule: test_correct_counts[candidate_rule] / float(
        test_correct_counts[candidate_rule] + test_incorrect_counts[candidate_rule]) for candidate_rule in
                            candidate_rules}
    # print(test_rule_confidence)
    # 排序
    sorted_test_confidence = sorted(test_rule_confidence.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_test_confidence[:5])
    # 输出规则信息
    print('**' * 20)
    for index in range(10):
        print("Rule #{0}".format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        premise_names = ",".join(get_moive_name(idx, movie_name_data) for idx in premise)
        conclusion_name = get_moive_name(conclusion, movie_name_data)
        print("Rule: if a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
        print(" - Train Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
        print(" - Test Confidence: {0:.3f}".format(test_rule_confidence[(premise, conclusion)]))
        print("")
        """


