import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    # 参考链接地址 https://blog.csdn.net/qq_40587575/article/details/81134464
    # 加载数据集 数据清洗 把第一列日期进行格式转换
    dataset = pd.read_csv('../data/leagues_NBA_2014_games_games.csv', parse_dates=['Date'])
    dataset.columns = ["Date", "Score Type", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]

    # 提取新特征
    # 找出获胜的球队
    dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
    y_true = dataset["HomeWin"].values
    # 由历史数据得出主场获胜的概率
    print("Home Win 百分比: {0:.1f}%".format(100 * dataset["HomeWin"].sum() / dataset["HomeWin"].count()))

    # 再增加两列 使用决策树 由上一次主场作战时结果推测这次这两只球队胜负情况
    dataset["HomeLastWin"] = False
    dataset["VisitorLastWin"] = False
    # 打印输出前5个数据
    # print(results.head())

    # 使用字典记录上一场球队的胜负结果 默认值为0 即False
    won_last = defaultdict(int)
    for index, row in dataset.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        row["HomeLastWin"] = won_last[home_team]
        row["VisitorLastWin"] = won_last[visitor_team]
        won_last[home_team] = row["HomeWin"]
        won_last[visitor_team] = not row["HomeWin"]
        dataset.iloc[index] = row

    # print(dataset.iloc[20:25])

    # 使用决策树
    # criterion 默认是'gini'基于基尼不纯度  'entropy' 基于信息熵
    clf = DecisionTreeClassifier(criterion='entropy', random_state=14)
    # 取新添加的两列作为特征输入
    X_previous_wins = dataset[['HomeLastWin', 'VisitorLastWin']].values
    # 获取交叉检验的平均正确率
    scores = cross_val_score(clf, X_previous_wins, y_true, scoring='accuracy', cv=5)
    mean_score = np.mean(scores) * 100
    # 准确率为 the accuracy is 57.80%
    print('the accuracy is %0.2f' % mean_score + '%')


    # 再次获取另外一个数据集 读取2013年球队的排名情况
    standings = pd.read_csv('../data/leagues_NBA_2013_standings_expanded-standings.csv')
    # print(standings['Team'] == 'Miami Heat')
    # print(standings[standings['Team'] == 'Miami Heat']['Rk'].values[0])

    # 创建一个新的特征值，主场球队的排名是否比对手排名高
    dataset['HomeTeamRanksHigher'] = 0
    for index, row in dataset.iterrows():
        home_team = row['Home Team']
        visitor_team = row['Visitor Team']
        if home_team == 'New Orleans Pelicans':  # 更换了名字的球队
            home_team = 'New Orleans Hornets'
        elif visitor_team == 'New Orleans Pelicans':
            visitor_team = 'New Orleans Hornets'
        # 获取主场 客场球队排名
        home_rank = standings[standings['Team'] == home_team]['Rk'].values[0]
        visitor_rank = standings[standings['Team'] == visitor_team]['Rk'].values[0]
        row['HomeTeamRanksHigher'] = int(home_rank > visitor_rank)
        dataset.iloc[index] = row

    X_home_higher = dataset[['HomeLastWin', 'VisitorLastWin', 'HomeTeamRanksHigher']].values
    cf1 = DecisionTreeClassifier(random_state=14)
    scores = cross_val_score(cf1, X_home_higher, y_true, scoring='accuracy', cv=5)
    mean_score = np.mean(scores) * 100
    print('增加球队去年排名后  the accuracy is %0.2f' % mean_score + '%')



