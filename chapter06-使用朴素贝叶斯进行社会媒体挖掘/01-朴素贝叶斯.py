import sys
import json
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn import naive_bayes, pipeline,feature_extraction, model_selection
# NLTK自然语言处理工具集
import nltk
# nltk.download('punkt')

"""
 鉴于书籍案例使用的Api 出现问题较多 弃掉 请看02
"""

if __name__ == '__main__':
    input_filename = '../data/python_tweets.json'
    classes_filename = '../data/python_classes.json'
    tweets_list = []
    with open(input_filename, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            tweets_list.append(json.loads(line))
    # tweets_list = pd.read_json(file_path, lines=True,)
    # print(tweets_list.head())
    tweet_sample = tweets_list
    labels_list = pd.read_json(classes_filename, orient='values')

    n_samples = min(len(labels_list), len(tweets_list))
    sample_tweets = [t['text'].lower() for t in tweets_list[:n_samples]]
    sample_labels = labels_list[:n_samples]pipo
    # print(sample_tweets)
    y_true = np.array(sample_labels)
    print("{:.1f}% have class 1".format(np.mean(y_true == 1) * 100))


    class NLTKBOW(TransformerMixin):

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return [{word: True for word in nltk.word_tokenize(document)} for document in X]

    pipeline = pipeline.Pipeline([('bag-of-words', NLTKBOW()),
                                 ('vectorizer', feature_extraction.DictVectorizer()),
                                 ('naive-bayes', naive_bayes.BernoulliNB())])
    scores = model_selection.cross_val_score(pipeline, sample_tweets, y_true, cv=5, scoring='f1')
    print("Score: {:.3f}".format(np.mean(scores)))


