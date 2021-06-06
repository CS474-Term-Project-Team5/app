from src.issue_holder import IssueHolder
from src.ne_extractor import NeExtractor
from utils.utils import bfs, capital
from src.keyword_extractor import KeywordExtractor
from env.env import SECOND_CLUSTERING_NUMBER, SUB_CLUSTER_PATH
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd


def main():
    dir_path = SUB_CLUSTER_PATH + r'/2015/'

    file_list = os.listdir(dir_path)

    for file in file_list:
        article = pd.read_csv(dir_path + file)
        keywordExtractor = KeywordExtractor()
        extracted_keyword = []

        print("=======================")
        print("[Issue]")
        issue = keywordExtractor.extract(article)
        print(capital(issue[0]))
        print()

        for cluster_idx in range(len(set(article.cluster))):
            cluster = article[article['cluster'] == cluster_idx]
            # extract corresponding cluster
            keyword_list = keywordExtractor.extract(cluster)[:30]
            extracted_keyword.append(" ".join(keyword_list))

        # vectorize using keyword n-grams
        vectorizer = CountVectorizer(max_features=1500, ngram_range=(
            3, 6), min_df=1, max_df=0.7, stop_words=stopwords.words('english'))
        X_count = vectorizer.fit_transform(extracted_keyword)

        # apply tfidf
        # X_tfidf = TfidfTransformer().fit_transform(X_count).toarray()

        # calculate similarity
        similarity = cosine_similarity(X_count, X_count)

        # extract corresponding (row, column) pairs
        indices = np.where(similarity > 0.04)
        index_row, index_col = indices
        similarity_result = zip(index_row, index_col)
        similarity_set = set()

        # join similar clusters
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

        # reclustering by similarity
        issue_holders = []
        neExtractor = NeExtractor()
        top_ranking = dict()
        for sub_cluster in cluster:
            sub_cluster = article[article['cluster'].isin(sub_cluster)]
            top_issue = keywordExtractor.extract(sub_cluster)
            ne_list = neExtractor.extract(sub_cluster)
            top_ranking[capital(top_issue[0])] = sub_cluster.shape[0]

            doc_num = sub_cluster.shape[0]
            person = [entities[0]
                      for entities in ne_list if entities[1] == "PERSON"]
            place = [entities[0]
                     for entities in ne_list if entities[1] == "LOC"]
            org = [entities[0] for entities in ne_list if entities[1] == "ORG"]

            issueHolder = IssueHolder(top_issue, doc_num, person, org, place)
            issue_holders.append(issueHolder)

        top_ranking = sorted(issue_holders, reverse=True,
                             key=lambda x: x.num)[:2]

        print("[Detailed Information(per envent)]")
        for topic in top_ranking:
            # print("({}: {})".format(topic.topic[0], topic.num), end=' ')
            print("Event: {}".format(capital(topic.topic[0])))
            print("\t- Person: {}".format(", ".join(topic.person[:5])))
            print("\t- Organization: {}".format(", ".join(topic.org[:5])))
            print("\t- Place: {}".format(", ".join(topic.place[:5])))
            print()
    print("=======================")


if __name__ == "__main__":
    main()
