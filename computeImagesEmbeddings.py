import pickle
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
import json as json
from PIL import Image

class ComputeImagesEmbeddings():
    def __init__(self, inputFileNameImages, inputFileNameNoImages, outputFileNameImages, outputFileNameNoImages, imageFolder):
        self.inputFileNameImages = inputFileNameImages
        self.inputFileNameNoImages = inputFileNameNoImages
        self.outputFileNameImages = outputFileNameImages
        self.outputFileNameNoImages = outputFileNameNoImages
        self.imageFolder = imageFolder

        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        

    def createMap(self):
        with open(self.inputFileNameImages, "r") as read_images_file:
            imagesFile = json.load(read_images_file)
        with open(self.inputFileNameNoImages, "r") as read_noimages_file:
            noImagesFile = json.load(read_noimages_file)
  
        steps = []
        images = []
        for i,imageobj in enumerate(imagesFile):
            image = Image.open(self.imageFolder + str(i) + '.jpg').convert('RGB')
            images.append(image)
            
        for step in noImagesFile:
            steps.append(step['description'])

        imageInput = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        imageInputEmbedding = self.model.get_image_features(**imageInput)
        imageInputEmbedding = F.normalize(imageInputEmbedding, p=2, dim=1)
        imageInputEmbedding =imageInputEmbedding.detach().cpu().numpy()

        textInput = self.processor(text=steps, return_tensors="pt", padding=True, truncation = True).to(self.device)
        textInputEmbedding = self.model.get_text_features(**textInput)
        textInputEmbedding = F.normalize(textInputEmbedding, p=2, dim=1) 
        textInputEmbedding = textInputEmbedding.detach().cpu().numpy() 
            
        # Write to file
        with open(self.outputFileNameImages + '.pickle', 'wb') as f:
            pickle.dump(imageInputEmbedding, f)

        with open(self.outputFileNameNoImages + '.pickle', 'wb') as f:
            pickle.dump(textInputEmbedding, f)

        return
    
    





