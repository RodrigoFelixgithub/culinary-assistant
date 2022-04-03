import spacy as spacy
import pprint as pp
from spacy import displacy
from pathlib import Path

class TextBasedSearch(): 
    def __init__(self, client, index_name):
        self.client = client
        self.index_name = index_name
        
    def queryOpenSearch(self, qtxt):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(qtxt)

        querytxt = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

        query_bm25 = {
        'size': 5,
        #  'fields': ['title'],
        #  'fields': ['id', 'contents'],
        #  'fields': ['id', 'contents', 'sentence_embedding'],
        'fields': ['recipeId', 'recipeTitle', 'recipeDescription'],
        '_source': '',
        'query': {
            'multi_match': {
            'query': querytxt,
            'fields': ['recipeTitle', 'recipeDescription']
            }
        }
        }

        response = self.client.search(
            body = query_bm25,
            index = self.index_name
        )
        print('\nQUERY: ' + querytxt)
        print('\nSearch results:')
        pp.pprint(response)