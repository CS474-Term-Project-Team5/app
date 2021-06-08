import ast
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from env.env import ARTICLE_DIR, SECOND_CLUSTERING_NUMBER, SUB_CLUSTER_PATH
from src.keyword_extractor import KeywordExtractor
from utils.utils import bfs, capital
from src.ne_extractor import NeExtractor
from src.issue_holder import IssueHolder

YEAR = [2015, 2016, 2017]


def main():
    data_path = ARTICLE_DIR + "/issue_vectors/" + "issue_vectors.csv"
    vector_data = pd.read_csv(data_path)
    for year in YEAR:
        dir_path = "./data/sub_cluster2" + "/{}/".format(year)

        file_list = os.listdir(dir_path)

        for file in file_list:
            article = pd.read_csv(dir_path + file)
            keywordExtractor = KeywordExtractor()
            cluster1 = int(file.split('.')[0][-1])

            print("=======================")
            print("[Issue]")
            issue = keywordExtractor.extract(article)
            print(capital(issue[0]))
            print()

            cluster = list(set(article.cluster_e))
            top2 = sorted(
                cluster, key=lambda x: article[article['cluster_e'] == x].shape[0], reverse=True)[:2]

            for origin in top2:
                print("[Issue]")
                issue = keywordExtractor.extract(
                    article[article['cluster_e'] == origin])
                print(capital(issue[0]))
                print()

                # TODO extract similarity vector
                relative_issue_vector = vector_data[vector_data.year == year]
                relative_issues_dict = dict()
                for i in range(len(relative_issue_vector)):
                    info = relative_issue_vector.iloc[i]
                    relative_issues_dict[i] = [
                        (info.cluster1, info.cluster2), info.vector]

                relative_issues = sorted(
                    relative_issues_dict.items(), key=lambda x: x[0])

                origin_vector = relative_issue_vector[(relative_issue_vector["cluster1"] == cluster1) & (
                    relative_issue_vector["cluster2"] == origin)].vector
                origin_vector = [ast.literal_eval(list(origin_vector)[0])]

                relative_vectors = [ast.literal_eval(
                    v[1][1]) for v in relative_issues]

                # print(origin_vector)
                # print(relative_vectors)

                similarity = cosine_similarity(
                    np.array(origin_vector), np.array(relative_vectors))

                relative_rank = [(i, similarity[0][i])
                                 for i in range(len(similarity[0]))]
                relative_rank = sorted(
                    relative_rank, key=lambda x: x[1], reverse=True)[:3]

                relative_cluster = [relative_issues_dict[r[0]][0]
                                    for r in relative_rank]

                # print(relative_cluster)

                neExtractor = NeExtractor()
                issue_holder_list = []
                for (c1, c2) in relative_cluster:
                    path = r"/root/app/data/sub_cluster2/{}/korea_herald_c_{}.csv".format(
                        year, c1)
                    df = pd.read_csv(path)
                    relative_cluster = df[df["cluster_e"] == c2]
                    doc_num = relative_cluster.shape[0]
                    person, org, place = neExtractor.extract_top_3(
                        relative_cluster)
                    top_issue = keywordExtractor.extract(relative_cluster)
                    issueHolder = IssueHolder(
                        top_issue, doc_num, person, org, place)
                    issue_holder_list.append(issueHolder)

                print("[Related-Issue Event]")
                print("{}".format(", ".join([holder.topic[0]
                      for holder in issue_holder_list])))
                print()

                print("[Detailed Information(per envent)]")
                for issueHolder in issue_holder_list:
                    print("Event: {}({})".format(
                        capital(issueHolder.topic[0]), issueHolder.num))
                    print(
                        "\t- Person: {}".format(", ".join(issueHolder.person[:3])))
                    print(
                        "\t- Organization: {}".format(", ".join(issueHolder.org[:3])))
                    print(
                        "\t- Place: {}".format(", ".join(issueHolder.place[:3])))
                    print()


def extract_issue_vector():
    features = ["year", "cluster1", "cluster2", "vector"]
    vector_data = []

    for year in YEAR:
        data_path = SUB_CLUSTER_PATH + "/{}/".format(year)
        file_list = os.listdir(data_path)

        for file in file_list:
            article = pd.read_csv(data_path + file)
            cluster1 = int(file.split('.')[0][-1])
            cluster2_list = list(set(article.cluster_e))

            for cluster2 in cluster2_list:
                sub_article = article[article.cluster_e == cluster2]
                try:
                    vector_list = sub_article.vector
                    vector_list = np.array([np.fromstring(np.array(vector))
                                            for vector in vector_list])
                    mean = vector_list.mean(axis=0)
                except Exception as e:
                    print(e)
                finally:
                    mean = [1, 2, 3, 4]
                    vector_data.append([year, cluster1, cluster2, mean])
                    # print(vector_info)

    result = pd.DataFrame(
        vector_data, columns=features)
    result.to_csv(ARTICLE_DIR + "/issue_vectors/" + "issue_vectors.csv")


if __name__ == "__main__":
    # extract_issue_vector()
    main()
