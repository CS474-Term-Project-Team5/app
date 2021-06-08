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

        for cluster_idx in range(len(set(article.cluster_e))):
            cluster = article[article['cluster_e'] == cluster_idx]
            # extract corresponding cluster
            keyword_list = keywordExtractor.extract_e(cluster)[:30]
            extracted_keyword.append(" ".join(keyword_list))

        # vectorize using keyword n-grams
        max_df = len(extracted_keyword) * \
            0.7 if len(extracted_keyword) > 1 else 1
        vectorizer = CountVectorizer(max_features=1500, ngram_range=(
            3, 6), min_df=1, max_df=max_df, stop_words=stopwords.words('english'))
        X_count = vectorizer.fit_transform(extracted_keyword)
        # apply tfidf
        # X_tfidf = TfidfTransformer().fit_transform(X_count).toarray()

        # calculate similarity
        similarity = cosine_similarity(X_count, X_count)
        # extract corresponding (row, column) pairs
        indices = np.where(similarity > 0.000004)
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
        issue_holders = {}
        neExtractor = NeExtractor()
        top2 = sorted(cluster, key=lambda x: article[article['cluster_e'].isin(
            x)].shape[0], reverse=True)
        print("[On-Issue Event]")
        for sub_cluster in top2:
            sub_cluster = article[article['cluster_e'].isin(sub_cluster)]
            months = sorted(list(set(sub_cluster.month)))
            for m in months:
                cluster_by_month = sub_cluster[sub_cluster.month == m]
                person, org, place = neExtractor.extract_top_3(
                    cluster_by_month)
                doc_num = cluster_by_month.shape[0]
                top_issue = keywordExtractor.extract(cluster_by_month)
                issueHolder = IssueHolder(
                    top_issue, doc_num, person, org, place)
                if doc_num >= sub_cluster.shape[0] / 12:
                    if m not in issue_holders:
                        issue_holders[m] = [issueHolder]
                    else:
                        issue_holders[m].append(issueHolder)

        for k, v in issue_holders.items():
            v = sorted(v, key=lambda x: x.num, reverse=True)
            issue_holders[k] = v[0]

        print("{}".format(
            " -> ".join([capital(holder.topic[0]) for holder in issue_holders.values()])))

        print("[Detailed Information(per envent)]")

        for holder in sorted(issue_holders.items(), key=lambda x: x[0]):
            holder = holder[1]
            # print("({}: {})".format(topic.topic[0], topic.num), end=' ')
            if capital(holder.topic[0]) == capital(issue[0]):
                print("Event: {}({})".format(
                    capital(holder.topic[1]), holder.num))
            else:
                print("Event: {}({})".format(
                    capital(holder.topic[0]), holder.num))
            print("\t- Person: {}".format(", ".join(holder.person[:3])))
            print("\t- Organization: {}".format(", ".join(holder.org[:3])))
            print("\t- Place: {}".format(", ".join(holder.place[:3])))
            print()

            # print detailed issues
            print("[Detailed Information(per envent)]")
            for holder in issue_holders:
                # print("({}: {})".format(topic.topic[0], topic.num), end=' ')
                print("Event: {}({})".format(
                    capital(holder.topic[0]), holder.num))
                print("\t- Person: {}".format(", ".join(holder.person[:3])))
                print("\t- Organization: {}".format(", ".join(holder.org[:2])))
                print("\t- Place: {}".format(", ".join(holder.place[:2])))
                print()
    print("=======================")


if __name__ == "__main__":
    main()
