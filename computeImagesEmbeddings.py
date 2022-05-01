import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json as json

class ComputeImagesEmbeddings():
    def __init__(self, inputFileNameImages, inputFileNameNoImages, outputFileNameImages, outputFileNameNoImages, tokenizer, model):
        self.inputFileNameImages = inputFileNameImages
        self.inputFileNameNoImages = inputFileNameNoImages
        self.outputFileNameImages = outputFileNameImages
        self.outputFileNameNoImages = outputFileNameNoImages
        
        # Load model from HuggingFace Hub
        self.tokenizer = tokenizer
        self.model = model

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

        
    #Mean Pooling - Take average of all tokens
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def createMap(self):
        with open(self.inputFileNameImages, "r") as read_images_file:
            imagesFile = json.load(read_images_file)
        with open(self.inputFileNameNoImages, "r") as read_noimages_file:
            noImagesFile = json.load(read_noimages_file)

        imageEmbeddings = []
        noImageEmbeddings = []

        for step in imagesFile:
            imageEmbeddings.append(self.encode(step['description'])[0].numpy()) 
            
        for step in noImagesFile:
            noImageEmbeddings.append(self.encode(step['description'])[0].numpy())
          
            
        # Write to file
        with open(self.outputFileNameImages + '.pickle', 'wb') as f:
            pickle.dump(imageEmbeddings, f)

        with open(self.outputFileNameNoImages + '.pickle', 'wb') as f:
            pickle.dump(noImageEmbeddings, f)

        return
    
    





