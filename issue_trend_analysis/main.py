
from env.env import ARTICLE_2015, ARTICLE_2016, ARTICLE_2017, FIRST_CLUSTERING_NUMBER, FIRST_RECLUSTERING_2015_SIM, FIRST_RECLUSTERING_2016_SIM, FIRST_RECLUSTERING_2017_SIM, RECLUSTERING_2015_PATH, RECLUSTERING_2016_PATH, RECLUSTERING_2017_PATH
from utils.utils import bfs
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from src.keyword_extraction import KeywordExtractor

YEAR = [2015, 2016, 2017]
ARTICLE = [ARTICLE_2015, ARTICLE_2016, ARTICLE_2017]
SIMILARITY = [FIRST_RECLUSTERING_2015_SIM,
              FIRST_RECLUSTERING_2016_SIM, FIRST_RECLUSTERING_2017_SIM]
SAVE_PATH = [RECLUSTERING_2015_PATH,
             RECLUSTERING_2016_PATH, RECLUSTERING_2017_PATH]


def main():
    for year in range(3):
        keywordExtractor = KeywordExtractor()
        keyword_2015 = keywordExtractor.extract(ARTICLE[year])

        # vectorize using keyword n-grams
        vectorizer = CountVectorizer(max_features=1500, ngram_range=(
            2, 5), min_df=1, max_df=0.7, stop_words=stopwords.words('english'))
        X_count = vectorizer.fit_transform(keyword_2015)

        # apply tfidf
        # X_tfidf = TfidfTransformer().fit_transform(X_count).toarray()

        # calculate similarity
        similarity = cosine_similarity(X_count, X_count)

        # extract corresponding (row, column) pairs
        indices = np.where(similarity > SIMILARITY[year])
        index_row, index_col = indices
        similarity_result = zip(index_row, index_col)
        similarity_set = set()
        for (r, c) in similarity_result:
            if (r != c):
                p = (r, c) if r < c else (c, r)
                similarity_set.add(p)

        similarity_result = list(similarity_set)

        # create sub cluster using bfs algorithm
        cluster_index = [i for i in range(FIRST_CLUSTERING_NUMBER)]
        cluster = []

        for i in cluster_index:
            if i in [e for sub in cluster for e in sub]:
                continue

            sub_cluster = []
            sub_cluster.append(i)
            cluster.append(bfs(sub_cluster, 0, similarity_result))

        # reclustering
        top_ranking = keywordExtractor.recluster(ARTICLE[year], cluster)

        top_ranking = sorted(top_ranking.items(), reverse=True,
                             key=lambda x: x[1])[:10]
        print("=======================")
        print("\t", YEAR[year])
        print("=======================")
        for topic in top_ranking:
            print(topic, end=' ')
        print()

        keywordExtractor.extract_file(ARTICLE[year], SAVE_PATH[year], cluster)


if __name__ == "__main__":
    main()
