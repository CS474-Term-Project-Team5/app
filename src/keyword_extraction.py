import ast
from utils.utils import capital, preprocessing
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from nltk.corpus import stopwords


class KeywordExtractor:
    def __init__(self) -> None:
        pass

    def extract(self, target_file: str) -> list:
        result_list = []
        article = pd.read_csv(target_file)

        for i in range(len(set(article.cluster))):
            cluster = article[article['cluster'] == i]
            keyword_set = cluster.keyword

            ne_set = cluster['ne']
            ne_set = [ne.lower()
                      for ne_list in ne_set for ne in ast.literal_eval(ne_list)]

            doc_num = len(keyword_set)
            keyword_set = [preprocessing(keyword.replace(
                ' ,', ' '), list(ne_set)) for keyword in keyword_set]

            # vectorize keywords
            vectorizer = CountVectorizer(max_features=1500, ngram_range=(
                2, 5), min_df=1, max_df=doc_num/3, stop_words=stopwords.words('english'))
            X_count = vectorizer.fit_transform(keyword_set).toarray()
            X_tfidf = TfidfTransformer().fit_transform(X_count).toarray()

            # extract top nth keyword
            result = pd.DataFrame(
                X_count, columns=vectorizer.get_feature_names())
            result = list(result.sum(axis=0).sort_values(
                ascending=False).keys()[:30])
            result_list.append(" ".join(result))
        return result_list

    def recluster(self, target_file: str, updated_cluster: list) -> dict:
        top = dict()
        article = pd.read_csv(target_file)

        for i in updated_cluster:
            cluster = article[article['cluster'].isin(i)]
            keyword_set = cluster.keyword

            ne_set = cluster['ne']
            ne_set = [ne.lower()
                      for ne_list in ne_set for ne in ast.literal_eval(ne_list)]

            doc_num = len(keyword_set)
            keyword_set = [preprocessing(keyword.replace(
                ' ,', ' '), list(ne_set)) for keyword in keyword_set]

            vectorizer = CountVectorizer(max_features=1500, ngram_range=(
                2, 5), min_df=1, max_df=doc_num/3, stop_words=stopwords.words('english'))
            X_count = vectorizer.fit_transform(keyword_set).toarray()
            X_tfidf = TfidfTransformer().fit_transform(X_count).toarray()

            result = pd.DataFrame(
                X_count, columns=vectorizer.get_feature_names())
            result = list(result.sum(axis=0).sort_values(
                ascending=False).keys()[:100])

            # print('=========={}:{}=========='.format(i, doc_num))
            # print('Topic:', result[:10])
            # print('Docs number:', doc_num)
            top[capital(result[0])] = doc_num
        return top
