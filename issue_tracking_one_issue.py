from utils.utils import capital, preprocessing
from env.env import SUB_CLUSTER_PATH
import os
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

dir_path = SUB_CLUSTER_PATH + r'/2015/'

file_list = os.listdir(dir_path)

for file in file_list:
    cluster = pd.read_csv(dir_path + file)
    doc_num = cluster.shape[0]

    ne_set = cluster["ne"]
    ne_set = [ne.lower()
              for ne_list in ne_set for ne in ast.literal_eval(ne_list)]

    keyword_set = cluster["keyword"]
    keyword_set = [preprocessing(keyword.replace(
        ' ,', ' '), list(ne_set)) for keyword in keyword_set]

    vectorizer = CountVectorizer(max_features=1500, ngram_range=(
        2, 5), min_df=1, max_df=doc_num/3, stop_words=stopwords.words('english'))
    X_count = vectorizer.fit_transform(keyword_set).toarray()

    result = pd.DataFrame(X_count, columns=vectorizer.get_feature_names())
    result = list(result.sum(axis=0).sort_values(ascending=False).keys()[:100])

    print()
    print('Topic:', capital(result[0]))
    print('Docs number:', doc_num)
    print()
