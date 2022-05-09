from os import truncate
import numpy as np
import json 
from PIL import Image
import requests
import spacy
import torch
from transformers import CLIPProcessor, CLIPModel

class ComputeClipOutput():
    def __init__(self, imageSimilarityMatrix, imagefile, noImagefile, imageFolder, recipesMap):
        self.recipesMap = recipesMap
        self.imageSimilarityMatrix = imageSimilarityMatrix
        self.imagefile = imagefile
        self.noImagefile = noImagefile
        self.imageFolder = imageFolder
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def getImage(self):
        with open(self.imagefile, "r") as read_images_file:
            imagesFile = json.load(read_images_file)
        with open(self.noImagefile, "r") as read_no_images_file:
            noImagesFile = json.load(read_no_images_file)
        with open(self.recipesMap, "r") as read_recipesMap:
            recipesMapMap = json.load(read_recipesMap)

        newStepsFile = self.noImagefile[:len(self.noImagefile) - 5] + 'Filled.json'
        for r,row in enumerate(self.imageSimilarityMatrix):
            nprow = np.array(row)
            indexes = np.argpartition(nprow, -10)[-10:]
            #top10 = row[indexes]
            #print(top10)
            if noImagesFile[r]['description'] is not None and len(noImagesFile[r]['description']) > 0 :
                sentence = noImagesFile[r]['description']
            elif noImagesFile[r]['title'] is not None :
                sentence = noImagesFile[r]['title']                
            else :
                sentence = ''
            
            images = []
            for i in indexes:
                #print(i)
                image = Image.open(self.imageFolder + str(i) + '.jpg').convert('RGB')
                images.append(image)

            VL_tokens = self.processor(text=sentence, images=images, return_tensors="pt", padding=True, truncation = True).to(self.device)
            outputs = self.model(**VL_tokens)
            logits_per_text = outputs.logits_per_text # this is the image-text similarity score
            probs = logits_per_text.softmax(dim=1) # we can take the softmax to get the label probabilities
            probabilities = probs.detach().cpu().numpy()[0]

            npprobabilities = np.array(probabilities)
            bestMatch = np.argpartition(npprobabilities, -1)[-1:]
            bestImage = bestMatch[0]
            image = indexes[bestImage]

            noImagesFile[r]['images'] = imagesFile[indexes[bestImage]]['images']
            recipesMapMap[noImagesFile[r]['recipeId']]['recipe']['instructions'][noImagesFile[r]['stepNumber']-1]['stepImages'] = imagesFile[indexes[bestImage]]['images']

            with open( newStepsFile, 'w') as f: #isto fica aqui embora demore mais tempo para o caso de algo acontecer temos o estado guardado no ficheiro
                json.dump(noImagesFile, f)

            print (r)

        newRecipesMap = self.recipesMap[:len(self.recipesMap) - 5] + 'WithImages.json'
        with open( newRecipesMap, 'w') as f: 
            json.dump(recipesMapMap, f)


        return image




