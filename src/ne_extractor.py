from utils.utils import capital
import spacy
import pandas as pd


class NeExtractor():
    def __init__(self):
        self.ne_type = ['ORG', 'PERSON', 'LOC']
        self.sp = spacy.load('en_core_web_sm')
        pass

    def extract(self, cluster: pd.DataFrame):
        rawtext = " ".join(cluster.keyword)
        ne = [self.sp(rawtext)]
        ne = [(e.text, e.lemma_, e.label_)
              for entities in ne for e in entities.ents]
        ne = set((capital(n[1]), n[2]) for n in ne if n[2] in self.ne_type)
        return list(ne)
