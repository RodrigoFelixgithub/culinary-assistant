from os import truncate
import json 
import numpy as np


class GetImage():
    def __init__(self, imageSimilarityMatrix, imagefile, noImagefile, imageFolder, recipesMap):
        self.recipesMap = recipesMap
        self.imageSimilarityMatrix = imageSimilarityMatrix
        self.imagefile = imagefile
        self.noImagefile = noImagefile
        self.imageFolder = imageFolder

    def getImageFunc(self):
        with open(self.imagefile, "r") as read_images_file:
            imagesFile = json.load(read_images_file)
        with open(self.noImagefile, "r") as read_no_images_file:
            noImagesFile = json.load(read_no_images_file)
        with open(self.recipesMap, "r") as read_recipesMap:
            recipesMapMap = json.load(read_recipesMap)

        newStepsFile = self.noImagefile[:len(self.noImagefile) - 5] + 'Filled.json'
        for r,row in enumerate(self.imageSimilarityMatrix):
            nprow = np.array(row)
            max_index = np.argpartition(nprow, -1)[-1:][0]            


            
            # if noImagesFile[r]['description'] is not None and len(noImagesFile[r]['description']) > 0 :
            #     sentence = noImagesFile[r]['description']
            # elif noImagesFile[r]['title'] is not None :
            #     sentence = noImagesFile[r]['title']                
            # else :
            #     sentence = ''
            
            # images = Image.open(self.imageFolder + str(max_index) + '.jpg').convert('RGB')


            noImagesFile[r]['images'] = imagesFile[max_index]['images']
            recipesMapMap[noImagesFile[r]['recipeId']]['recipe']['instructions'][noImagesFile[r]['stepNumber']-1]['stepImages'] = imagesFile[max_index]['images']

            print (r)

        with open( newStepsFile, 'w') as f: 
            json.dump(noImagesFile, f)

        newRecipesMap = self.recipesMap[:len(self.recipesMap) - 5] + 'WithImages.json'
        with open( newRecipesMap, 'w') as f: 
            json.dump(recipesMapMap, f)





