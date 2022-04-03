import json as json
import spacy
from spacy import displacy
from pathlib import Path

class IndexRecipes():
    def __init__(self, fileName, client, index_name):
        self.fileName = fileName
        self.client = client
        self.index_name = index_name

    def indexRecipes(self):
        nlp = spacy.load("en_core_web_sm")
        with open(self.fileName, "r") as read_file:
            recipes = json.load(read_file)

        for recipeId in recipes:
            recipeTitleDoc = nlp(recipes[recipeId]['recipe']['displayName'])
            recipeTitleString = ' '.join([token.lemma_ for token in recipeTitleDoc if not token.is_stop and token.is_alpha])
            if recipes[recipeId]['recipe']['description'] == None :
                recipeDescriptionString = None
            else:
                recipeDescriptionDoc = nlp(recipes[recipeId]['recipe']['description'])
                recipeDescriptionString = ' '.join([token.lemma_ for token in recipeDescriptionDoc if not token.is_stop and token.is_alpha])
            doc = {
            'recipeId': recipeId,
            'title': recipeTitleString,
            'description': recipeDescriptionString,
            }
            resp = self.client.index(index=self.index_name, id=recipeId, body=doc)
            print(resp['result'])        

        return