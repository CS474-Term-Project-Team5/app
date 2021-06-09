import ast
import pandas as pd
import pickle
import torch
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from env.env import ARTICLE_DIR, SECOND_CLUSTERING_NUMBER, SUB_CLUSTER_PATH
from src.keyword_extractor import KeywordExtractor
from utils.utils import bfs, print_event, print_issue, capital
from src.ne_extractor import NeExtractor
from src.issue_holder import IssueHolder
from src.event_holder import EventHolder

YEAR = [2015]


def main():
    for year in YEAR:
        dir_path = "./data/sub_cluster" + "/{}/".format(year)
        file_list = [i for i in os.listdir(dir_path) if ".csv" in i]
        vec_list = [i for i in os.listdir(dir_path) if ".txt" in i]
        file_list = [(file_list[i], vec_list[i]) for i in range(len(file_list))]

        issue = {}

        for file,vec_file in file_list:
            article = pd.read_csv(dir_path + file)
            keywordExtractor = KeywordExtractor()
            neExtractor = NeExtractor()
            extracted_keyword = []

            with open (dir_path + vec_file, 'rb') as lf:
                vec_list = pickle.load(lf)

            article['vec_e'] = vec_list
            issue_vec = torch.stack(vec_list).mean(dim = 0)
            keywordExtractor = KeywordExtractor()
            cluster1 = int(file.split('.')[0][-1])

            issue_num = article.shape[0]
            issue_title = [keywordExtractor.extract(article)[0]]

            issueHolder = IssueHolder(
                issue_title, issue_num, issue_vec)

            for cluster_idx in range(len(set(article.cluster_e))):
                cluster = article[article['cluster_e'] == cluster_idx]
                # extract corresponding cluster
                keyword_list = keywordExtractor.extract_e(cluster)[:30]
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


            print(cluster)
            event = []

            for sub_cluster in cluster:

                sub_cluster = article[article['cluster_e'].isin(sub_cluster)]
                doc_num = sub_cluster.shape[0]
                if issue_num / 20 <= doc_num:
                    person, org, place = neExtractor.extract_top_3(sub_cluster)
                    top_issue = keywordExtractor.extract(sub_cluster)
                    if top_issue[0] == issue_title:
                        top_issue = top_issue[1:4]
                    else:
                        top_issue = top_issue[0:3]

                    event_vec = torch.stack(sub_cluster[:]['vec_e'].tolist()).mean(dim=0)
                    eventHolder = EventHolder(
                        top_issue, doc_num, person, org, place, event_vec)
                    event.append(eventHolder)

            issueHolder.add_event(event)
            issue[cluster1] = issueHolder

        for i in issue:
            print_issue(issue[i])
            top_3 = []
            for j in issue:
                if i == j:
                    pass
                else:
                    for e in issue[j].event:
                        if len(top_3) < 3:
                            print(e.vec, issue[i].vec)
                            dist = torch.dist(e.vec, issue[i].vec, p=2.0)
                            top_3.append((e,dist))
                            top_3 = sorted(top_3,key=lambda x:x[1])
                        else:
                            dist = torch.dist(e.vec, issue[i].vec, p=2.0)
                            if top_3[-1][1] > dist:
                                top_3[-1] = (e,dist)
                            top_3 = sorted(top_3,key=lambda x:x[1])

            print("{}".format(
                "  ".join([capital(holder[0].topic[0]) for holder in top_3])))
            for e in top_3:
                print_event(e[0])



if __name__ == "__main__":
    # extract_issue_vector()
    main()