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

    # remove president name
    # document = re.sub(r'president', ' ', document)
    document = re.sub(r'south korea', ' ', document)
    document = re.sub(r'north korea', ' ', document)
    document = re.sub(r'seoul', ' ', document)

    document = re.sub(r'president park', ' ', document)
    document = re.sub(r'korean geun', ' ', document)
    document = re.sub(r'geun', ' ', document)
    document = re.sub(r'geun hye', ' ', document)
    document = re.sub(r'president hye', ' ', document)

    document = re.sub(r'president moon', ' ', document)
    document = re.sub(r'moon jae', ' ', document)
    document = re.sub(r'korean jae', ' ', document)
    document = re.sub(r'korea', ' ', document)
    document = re.sub(r'korean', ' ', document)

    # document = re.sub(r'us donald', ' ', document)
    document = document.replace('us donald', ' ')
    document = document.replace('donald', ' ')
    document = document.replace('trump', ' ')
    document = document.replace('us', ' ')

    document = re.sub(r'kim jong', ' ', document)

    for entity in ne_set:
        document = document.replace(" {} ".format(entity), ' ')

    document = re.sub(r'\s+', ' ', document)
    return document


def capital(string):
    # print(string)
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
                if c not in sub_cluster:
                    sub_cluster.append(c)
            elif sub_cluster[i] == c:
                if r not in sub_cluster:
                    sub_cluster.append(r)
    return bfs(sub_cluster, next_start, data)
