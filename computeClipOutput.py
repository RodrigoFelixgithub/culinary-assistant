import numpy as np
import json 
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

class ComputeClipOutput():
    def __init__(self, imageSimilarityMatrix, imagefile, noImagefile, imageFolder):
        self.imageSimilarityMatrix = imageSimilarityMatrix
        self.imagefile = imagefile
        self.noImagefile = noImagefile
        self.imageFolder = imageFolder
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def getImage(self):
        with open(self.imagefile, "r") as read_images_file:
            imagesFile = json.load(read_images_file)
        with open(self.noImagefile, "r") as read_no_images_file:
            noImagesFile = json.load(read_no_images_file)


        for r,row in enumerate(self.imageSimilarityMatrix):
            nprow = np.array(row)
            indexes = np.argpartition(nprow, -10)[-10:]
            #top10 = row[indexes]
            #print(top10)
            sentence = noImagesFile[r]['title'] if noImagesFile[r]['title'] is not None else noImagesFile[r]['description']
            images = []
            for i in indexes:
                #print(i)
                image = Image.open(self.imageFolder + str(i) + '.jpg')
                images.append(image)



            VL_tokens = self.processor(text=sentence, images=images, return_tensors="pt", padding=True)

            outputs = self.model(**VL_tokens)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
            probs




