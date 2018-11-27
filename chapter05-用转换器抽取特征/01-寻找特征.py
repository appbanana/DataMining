import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import feature_selection
from sklearn import tree
from sklearn import model_selection
from sklearn import decomposition

if __name__ == '__main__':
    file_path = '../data/adult.data'
    adult = pd.read_csv(file_path,
                        header=None,
                        names=["Age", "Work-Class", "fnlwgt", "Education", "Education-Num",
                               "Marital-Status", "Occupation","Relationship", "Race", "Sex",
                               "Capital-gain", "Capital-loss", "Hours-per-week", "Native-Country", "Earnings-Raw"],
                        memory_map=True)
    # 去除重复的数据
    adult.drop_duplicates()
    # 去除某行 所有数据为None的数据
    adult.dropna(how='all', inplace=True)
    print(adult.head())
    print('\n')
    # print(adult['Hours-per-week'].describe())
    # unique去重 把所有的工作地罗列出来
    # print(adult['Work-Class'].unique())
    """
    [' State-gov' ' Self-emp-not-inc' ' Private' ' Federal-gov' ' Local-gov' ' ?' ' Self-emp-inc' ' Without-pay' ' Never-worked']
    """
    # 受教育年限统计 降序输出
    # print(adult['Education-Num'].value_counts())
    """
    9     10501
    10     7291
    13     5355
    14     1723
    11     1382
    7      1175
    12     1067

    """
    #######################################################
    # 特征选择
    # 1. scikit-learn中的VarianceThreshold转换器可用来删除特征值方差达不到最低标准的特征
    # 官方参考地址 https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    """
    X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
    selector = VarianceThreshold(threshold=0.0) #阈值默认为0
    selector.fit_transform(X)
    seletor.variances_   # 打印每一列的特征值方差 array([0.        , 0.22222222, 2.88888889, 0.        ])
    """

    # 选择最佳特征  SelectKBest返回k个最佳特征值   SelectPercentile 返回最佳的前r%个特征
    X = adult[['Age', 'Education-Num', 'Capital-gain', 'Capital-loss', 'Hours-per-week']].values
    # 我擦 >50k前面竟然有个空格
    # y = (adult['Earnings-Raw'] == '>50K').values
    y = (adult["Earnings-Raw"] == ' >50K').values
    # print(X)
    # SelectKBest和SelectPercentile使用方法很相似
    # SelectKBest
    transformer = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=3)
    # SelectPercentile
    # transformer = feature_selection.SelectPercentile(score_func=feature_selection.chi2, percentile=3)
    Xt_chi2 = transformer.fit_transform(X, y)
    # 根据下面的打分 显然第一列 第三列 第四列相关性更好
    # print(transformer.scores_)
    """
    [8.60061182e+03 2.40142178e+03 8.21924671e+07 1.37214589e+06 6.47640900e+03]
    """

    # 使用皮尔逊计算相关性
    """
    p值为1到1之间的任意值。值为1，表示两个变量之间绝对正相关，值为1， 1 绝对负相关，
    即一个变量值越大，另一个变量值就越小，反之亦然。这样的特征 确实能反应两个变量之间的关系，
    但是根据大小进行排序，这些值因为是负数而 排在后面，可能会被舍弃不用
    """
    def multivariate_pearsonr(X, y):
        """
        :param X:
        :param y:
        :return: 返回包含皮尔逊相关系数和p值的元组。
        """
        scores, pvalues = [], []
        for column in range(X.shape[1]):
            # 遍历每一行中 所有的列
            cur_score, cur_p = pearsonr(X[:, column], y)
            scores.append(cur_score)
            pvalues.append(cur_p)
        return (np.array(scores), np.array(pvalues))

    # 在此使用转换器 使用皮尔逊方法得到第一列 第二列 第五列相关性最好
    transformer = feature_selection.SelectKBest(score_func=multivariate_pearsonr, k=3)
    Xt_pearson = transformer.fit_transform(X, y)
    # print(transformer.scores_)
    """
    [0.2340371  0.33515395 0.22332882 0.15052631 0.22968907]
    """

    # 使用分类器看看那个效果更好
    clf = tree.DecisionTreeClassifier(random_state=14)
    score_chi2 = model_selection.cross_val_score(clf, Xt_chi2, y, scoring='accuracy', cv=5)
    score_pearson = model_selection.cross_val_score(clf, Xt_pearson, y, scoring='accuracy', cv=5)
    score_mean_chi2 = np.mean(score_chi2)
    score_mean_pearson = np.mean(score_pearson)

    print('score_chi2 = %.2f' % score_mean_chi2)
    print('score_pearson = %.2f' % score_mean_pearson)
    """
    score_chi2 = 0.83
    score_pearson = 0.77
    """
    #######################################################
    # PCA 主成分分析
    pca = decomposition.PCA(n_components=5)
    Xd = pca.fit_transform(X)
    clf = tree.DecisionTreeClassifier()
    score_pca = model_selection.cross_val_score(clf, Xd, y, cv=5, scoring='accuracy')
    score_mean_pca = np.mean(score_pca)
    # print(pca.explained_variance_ratio_)
    print('score_mean_pca = %.2f' % score_mean_pca)
    """
    score_mean_pca = 0.82
    """

    #######################################################
    # 创建自己的转换器



