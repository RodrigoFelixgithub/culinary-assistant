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

            # Tokenization of the recipe title and description, using the recipes file
            recipeTitleDoc = nlp(recipes[recipeId]['recipe']['displayName'])
            recipeTitleString = ' '.join([token.lemma_ for token in recipeTitleDoc if not token.is_stop and token.is_alpha])
            if recipes[recipeId]['recipe']['description'] == None :
                recipeDescriptionString = None
            else:
                recipeDescriptionDoc = nlp(recipes[recipeId]['recipe']['description'])
                recipeDescriptionString = ' '.join([token.lemma_ for token in recipeDescriptionDoc if not token.is_stop and token.is_alpha])
            
            # Get embeddings from the embeddings file to use in the indexes
            sentence_embedding_title = numpy.asarray(embeddings[recipeId]['title_embedding'])
            if recipeDescriptionString == None :
                doc = {
                    'recipeId': recipeId,
                    'title': recipeTitleString,
                    'sentence_embedding_title': sentence_embedding_title,
                }
            else:
                doc = {
                    'recipeId': recipeId,
                    'title': recipeTitleString,
                    'description': recipeDescriptionString,
                    'sentence_embedding_title': sentence_embedding_title,
                    'sentence_embedding_description': numpy.asarray(embeddings[recipeId]['description_embedding'])
                }

            resp = self.client.index(index=self.index_name, id=recipeId, body=doc)
            print(resp['result'])        

        return