import pprint as pp
from pathlib import Path
import torch
import torch.nn.functional as F

class EmbeddingBasedSearch(): 
    def __init__(self, client, index_name):
        self.client = client
        self.index_name = index_name

    #Encode text
    def encode(self, texts):
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        
        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
            
    def queryOpenSearch(self, qtxt):
        query_emb = self.encode(qtxt)

        query_denc = {
        'size': 5,
        #  '_source': ['doc_id', 'contents', 'sentence_embedding'],
        #  '_source': ['doc_id', 'contents'],
        '_source': ['doc_id'],
        "query": {
                "knn": {
                "sentence_embedding": {
                    "vector": query_emb[0].numpy(),
                    "k": 2
                }
                }
            }
        }

        response = self.client.search(
            body = query_denc,
            index = self.index_name
        )

        print('\nSearch results:')
        pp.pprint(response)

    