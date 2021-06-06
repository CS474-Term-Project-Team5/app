from utils.utils import bfs
from src.keyword_extraction import KeywordExtractor
from env.env import SECOND_CLUSTERING_NUMBER, SUB_CLUSTER_PATH
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


def main():
    dir_path = SUB_CLUSTER_PATH + r'/2015/'

    file_list = os.listdir(dir_path)

    for file in file_list:
        keywordExtractor = KeywordExtractor()
        extracted_keyword = keywordExtractor.extract(dir_path + file)

        # vectorize using keyword n-grams
        vectorizer = CountVectorizer(max_features=1500, ngram_range=(
            2, 5), min_df=1, max_df=0.7, stop_words=stopwords.words('english'))
        X_count = vectorizer.fit_transform(extracted_keyword)

        # apply tfidf
        # X_tfidf = TfidfTransformer().fit_transform(X_count).toarray()

        # calculate similarity
        similarity = cosine_similarity(X_count, X_count)

        # extract corresponding (row, column) pairs
        indices = np.where(similarity > 0.03)
        index_row, index_col = indices
        similarity_result = zip(index_row, index_col)
        similarity_set = set()
        for (r, c) in similarity_result:
            if (r != c):
                p = (r, c) if r < c else (c, r)
                similarity_set.add(p)

        similarity_result = list(similarity_set)

        # create sub cluster using bfs algorithm
        cluster_index = [i for i in range(SECOND_CLUSTERING_NUMBER)]
        cluster = []

        for i in cluster_index:
            if i in [e for sub in cluster for e in sub]:
                continue

            sub_cluster = []
            sub_cluster.append(i)
            cluster.append(bfs(sub_cluster, 0, similarity_result))

        # reclustering
        top_ranking = keywordExtractor.recluster(dir_path + file, cluster)

        top_ranking = sorted(top_ranking.items(), reverse=True,
                             key=lambda x: x[1])[:2]
        print("=======================")
        for topic in top_ranking:
            print(topic, end=' ')
        print()
    print("=======================")


if __name__ == "__main__":
    main()
