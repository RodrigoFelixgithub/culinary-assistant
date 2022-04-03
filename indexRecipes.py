import json as json
import spacy
from spacy import displacy
from pathlib import Path
import numpy as numpy

class IndexRecipes():
    def __init__(self, recipesFile, embeddingsFile, client, index_name):
        self.recipesFile = recipesFile
        self.embeddingsFile = embeddingsFile
        self.client = client
        self.index_name = index_name

    def indexRecipes(self):
        nlp = spacy.load("en_core_web_sm")
        with open(self.recipesFile, "r") as read_file:
            recipes = json.load(read_file)
        with open(self.embeddingsFile, "r") as read_file:
            embeddings = json.load(read_file)

        for recipeId in recipes:

            #tokenization and lemmatization of the recipe title and description, in this part we use the recipes file
            recipeTitleDoc = nlp(recipes[recipeId]['recipe']['displayName'])
            recipeTitleString = ' '.join([token.lemma_ for token in recipeTitleDoc if not token.is_stop and token.is_alpha])
            if recipes[recipeId]['recipe']['description'] == None :
                recipeDescriptionString = None
            else:
                recipeDescriptionDoc = nlp(recipes[recipeId]['recipe']['description'])
                recipeDescriptionString = ' '.join([token.lemma_ for token in recipeDescriptionDoc if not token.is_stop and token.is_alpha])
            
            #get the embeddings from the embeddings file and use them in the indexes
            sentence_embedding_title = numpy.asarray(embeddings[recipeId]['title_embedding'])
            sentence_embedding_description = numpy.asarray(embeddings[recipeId]['description_embedding'])
            sentence_embedding_title_trained = numpy.asarray(embeddings[recipeId]['title_embedding'])

            doc = {
            'recipeId': recipeId,
            'title': recipeTitleString,
            'description': recipeDescriptionString,
            'sentence_embedding_title': sentence_embedding_title,
            'sentence_embedding_description': sentence_embedding_description,
            'sentence_embedding_title_trained':sentence_embedding_title_trained
            }
            resp = self.client.index(index=self.index_name, id=recipeId, body=doc)
            print(resp['result'])        

        return