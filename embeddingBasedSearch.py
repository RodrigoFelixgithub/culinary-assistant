from transformers import AutoTokenizer, AutoModel
import pprint as pp
from pathlib import Path
import torch
import torch.nn.functional as F

class EmbeddingBasedSearch(): 
    def __init__(self, client, index_name):
        self.client = client
        self.index_name = index_name
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")
   
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
                    "sentence_embedding_title": {
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

    