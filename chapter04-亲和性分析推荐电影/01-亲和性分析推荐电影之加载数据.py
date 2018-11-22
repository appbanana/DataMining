from datetime import datetime
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
    # print('---favorable_reviews_by_users---' * 3)
    temp = rating[['MovieID', 'Favorable']].groupby('MovieID')
    # for k, v in rating[['MovieID', 'Favorable']].groupby('MovieID').sum:
    #     print(k, list(v))
    num_favorable_by_movie = rating[['MovieID', 'Favorable']].groupby('MovieID').sum()
    print(num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5])
    print('\n')
    print(num_favorable_by_movie.iterrows())

    # 生成初始的频繁项集 项集大于最小支持度则为频繁项集
    frequent_itemsets = {}
    min_support = 50
    # 将下面代码转化为集合表达式
    # for MovieID, row in num_favorable_by_movie.iterrows():
    #     if row['Favorable'] >= min_support:
    #         print((frozenset((MovieID,)),row["Favorable"]))
    frequent_itemsets[1] = dict((frozenset((moive_id,)), row['Favorable']) for moive_id, row in
                            num_favorable_by_movie.iterrows() if row['Favorable'] >= min_support)
    print(frequent_itemsets)

    print('\n')

    sys.stdout.flush()
    for k in range(2, 4):
        # Generate candidates of length k, using the frequent itemsets of length k-1
        # Only store the frequent itemsets
        print(frequent_itemsets[k-1])
        print('\n')

        cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k-1],
                                                       min_support)
        if len(cur_frequent_itemsets) == 0:
            print("Did not find any frequent itemsets of length {}".format(k))
            sys.stdout.flush()
            break
        else:
            print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
            #print(cur_frequent_itemsets)
            sys.stdout.flush()
            frequent_itemsets[k] = cur_frequent_itemsets
    # We aren't interested in the itemsets of length 1, so remove those
    del frequent_itemsets[1]
