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

    def checkNegative(self, keyword):
        return "free" in keyword or "no" in keyword

    def cleanNegativeWord(self, keyword):
        keyword = keyword.replace("free","").strip()
        keyword = keyword.replace("no ","").strip()
        return keyword

    def indexRecipes(self):
        nlp = spacy.load("en_core_web_sm")

        with open(self.recipesFile, "r") as read_file:
            recipes = json.load(read_file)

        with open(self.embeddingsFile, "r") as read_file:
            embeddings = json.load(read_file)

        for recipeId in recipes:

            keywordsPositive = []
            keywordsNegative = []
            for keyword in recipes[recipeId]['keywords']:
                keywordDoc = nlp(keyword)
                keywordString = ' '.join([token.lemma_ for token in keywordDoc if not token.is_stop and token.is_alpha])
                if self.checkNegative(keyword):
                    keywordCleaned = self.cleanNegativeWord(keywordString)
                    keywordsNegative.append(keywordCleaned)
                else:
                    keywordsPositive.append(keywordString)

            time = recipes[recipeId]['recipe']['totalTimeMinutes']

            ingredients = recipes[recipeId]['recipe']['ingredients']
            ingredientsArray = []
            for i in ingredients:
                ingredient = i['ingredient']
                ingredientsArray.append(ingredient)
                
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
                    'ingredients': ingredientsArray,
                    'description': recipeTitleString,
                    'time': time,
                    'positive_Keywords': keywordsPositive,
                    'negative_Keywords': keywordsNegative,
                    'sentence_embedding_title': sentence_embedding_title,
                    'sentence_embedding_description': sentence_embedding_title
                }
            else:
                doc = {
                    'recipeId': recipeId,
                    'title': recipeTitleString,
                    'ingredients': ingredientsArray,
                    'time': time,
                    'positive_Keywords': keywordsPositive,
                    'negative_Keywords': keywordsNegative,
                    'description': recipeDescriptionString,
                    'sentence_embedding_title': sentence_embedding_title,
                    'sentence_embedding_description': numpy.asarray(embeddings[recipeId]['description_embedding'])
                }

            resp = self.client.index(index=self.index_name, id=recipeId, body=doc)
            #print(resp['result'])        

        return