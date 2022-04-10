from transformers import AutoTokenizer, AutoModel
import pprint as pp
from pathlib import Path
import torch
import torch.nn.functional as F
import spacy as spacy
from spacy import displacy


class Search():
    def __init__(self, client, index_name, tokenizer, model):
        self.client = client
        self.index_name = index_name
        # Load model from HuggingFace Hub
        self.tokenizer = tokenizer
        self.model = model

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Encode text
    def encode(self, texts):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def filtersfunc(self, ings):
        ingredients = []
        if ings is not None:
            for i in ings:
                ingredients.append({"term": {"ingredients": i}})
        return ingredients

    def matchesfunc(self, keys, flag):
        nlp = spacy.load("en_core_web_sm")
        keywords = []
        if keys is not None:
            if flag:
                for k in keys:
                    keywordsPositiveDoc = nlp(k)
                    keywordsPositiveTxt = ' '.join([token.lemma_ for token in keywordsPositiveDoc if not token.is_stop and token.is_alpha])
                    keywords.append({"match": {"positive_Keywords": keywordsPositiveTxt}})
            else:
                for k in keys:
                    keywordsNegativeDoc = nlp(k)
                    keywordsNegativeTxt = ' '.join([token.lemma_ for token in keywordsNegativeDoc if not token.is_stop and token.is_alpha])
                    keywords.append({"match": {"negative_Keywords": keywordsNegativeTxt}})
        return keywords

    def timeFunc(self, time):
        rangeArray = []
        if time is not None:
            rangeArray.append({"range": { "time" : {"lte" : time}}})
        return rangeArray

    def queryOpenSearch(self, qtxt, nresults, ingsWanted, ingsNotWanted, keywordsPositive, keywordsNegative, time):
        query_emb = self.encode(qtxt)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(qtxt)
        querytxt = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        K_TRESHOLD = 10

        query_denc = {
            'size': nresults,
            #  '_source': ['doc_id', 'contents', 'sentence_embedding'],
            #  '_source': ['doc_id', 'contents'],
            '_source': '',
            'fields': ['recipeId', 'title', 'description', 'negative_Keywords'],
            "query": {
                'bool': {
                    'must': [
                        {
                            "knn": {
                                "sentence_embedding_title": {
                                    "vector": query_emb[0].numpy(),
                                    "k": nresults if nresults >= K_TRESHOLD else K_TRESHOLD
                                }
                            }
                        },
                        {
                            "knn": {
                                "sentence_embedding_description": {
                                    "vector": query_emb[0].numpy(),
                                    "k": nresults if nresults >= K_TRESHOLD else K_TRESHOLD
                                }
                            }
                        },
                        {
                            'multi_match': {
                                'query': querytxt,
                                'fields': ['title', 'description']
                            }
                        }
                    ],
                    'should': [*self.matchesfunc(keywordsPositive, True),*self.matchesfunc(keywordsNegative, False),*self.timeFunc(time)],
                    "filter": {
                        'bool': {
                            'must': self.filtersfunc(ingsWanted),
                            'must_not': self.filtersfunc(ingsNotWanted),
                        }
                    }
                },
            }
        }

        response = self.client.search(
            body=query_denc,
            index=self.index_name
        )
        

        print('\nSearch results:')
        return response