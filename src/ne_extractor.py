from utils.utils import capital
import spacy
import pandas as pd


class NeExtractor():
    def __init__(self):
        self.ne_type = ['ORG', 'PERSON', 'LOC']
        self.sp = spacy.load('en_core_web_sm')
        pass

    def extract(self, cluster: pd.DataFrame):
        rawtext = " ".join(cluster.description)
        ne = [self.sp(rawtext)]
        ne = [(e.text, e.lemma_, e.label_)
              for entities in ne for e in entities.ents]
        ne = set((capital(n[1]), n[2]) for n in ne if n[2] in self.ne_type)
        return list(ne)

    def extract_top_3(self, cluster: pd.DataFrame):
        rawtext = " ".join(cluster.body)
        ne = [self.sp(rawtext)]
        ne = [(e.text, e.lemma_, e.label_)
              for entities in ne for e in entities.ents]
        count_o = {}
        count_p = {}
        count_l = {}
        for n in ne:
            if n[2] not in self.ne_type:
                pass
            else:
                if n[2] == self.ne_type[0]:
                    count = count_o
                elif n[2] == self.ne_type[1]:
                    count = count_p
                else:
                    count = count_l

                if n[1] in count:
                    count[n[1]] += 1
                else:
                    count[n[1]] = 1

        count_o = [capital(n[0]) for n in sorted(count_o.items(),key=(lambda x:x[1]), reverse=True)[:3]]
        count_l = [capital(n[0]) for n in sorted(count_l.items(),key=(lambda x:x[1]), reverse=True)[:3]]
        count_p = [capital(n[0]) for n in sorted(count_p.items(),key=(lambda x:x[1]), reverse=True)[:3]]

        return (count_p, count_o, count_l)
