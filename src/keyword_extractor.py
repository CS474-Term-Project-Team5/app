import ast
from utils.utils import capital, preprocessing
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


class KeywordExtractor:
    def __init__(self, ngram: tuple = (2, 5)) -> None:
        self.ngram = ngram

    def extract(self, cluster: pd.DataFrame) -> list:
        keyword_set = cluster.keyword

        ne_set = cluster['ne']
        ne_set = [ne.lower()
                  for ne_list in ne_set for ne in ast.literal_eval(ne_list)]

        doc_num = len(keyword_set)
        keyword_set = [preprocessing(keyword.replace(
            ' ,', ' '), list(ne_set)) for keyword in keyword_set]

        # vectorize keywords
        max_df = doc_num * 0.7 if doc_num > 1 else 1
        vectorizer = CountVectorizer(max_features=1500, ngram_range=self.ngram,
                                     min_df=1, max_df=max_df, stop_words=stopwords.words('english'))
        X_count = vectorizer.fit_transform(keyword_set).toarray()

        # extract top nth keywords
        result = pd.DataFrame(
            X_count, columns=vectorizer.get_feature_names())
        result = list(result.sum(axis=0).sort_values(
            ascending=False).keys())
        return result

    def extract_file(self, article: pd.DataFrame, save_path: str, cluster: list) -> None:
        for i in cluster:
            article.loc[article['cluster'].isin(i), 'cluster'] = i[0]
            article.to_csv(save_path)
