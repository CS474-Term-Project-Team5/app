import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
stemmer = WordNetLemmatizer()


def preprocessing(rawtext: str, ne_set: set) -> str:
    document = rawtext
    document = re.sub(r'said', '', document)

    # Remove date
    dates = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
             'saturday', 'sunday', 'week', 'next', 'month', 'year']
    for date in dates:
        document = re.sub(r'{}'.format(date), '', document)

    # Remove publisher
    document = re.sub(r'korea herald', ' ', document)
    document = re.sub(r'history textbooks', ' ', document)
    document = re.sub(r'history textbook', ' ', document)

    for entity in ne_set:
        document = document.replace(" {} ".format(entity), ' ')

    document = re.sub(r'\s+', ' ', document)
    return document


def capital(string):
    string_list = string.split()
    string_list = [s.capitalize() for s in string_list]
    return ' '.join(string_list)


def bfs(sub_cluster: list, start: int, data: list) -> list:
    if (len(sub_cluster) == start):
        return sub_cluster

    next_start = len(sub_cluster)
    for i in range(start, len(sub_cluster)):
        for (r, c) in data:
            if sub_cluster[i] == r:
                if not c in sub_cluster:
                    sub_cluster.append(c)
            elif sub_cluster[i] == c:
                if not r in sub_cluster:
                    sub_cluster.append(r)
    return bfs(sub_cluster, next_start, data)
