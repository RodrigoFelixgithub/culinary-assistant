import json as json

import requests


class CreateStepsMaps():
    def __init__(self, inputFileName, imagesFileName, noImagesFileName):
        self.inputFileName = inputFileName
        self.imagesFileName = imagesFileName
        self.noImagesFileName = noImagesFileName

    def createMaps(self):
        with open(self.inputFileName, "r") as read_file:
            data = json.load(read_file)

        imageSteps = []
        noImageSteps = []
        ctr = 0
        for i in data:
            for j in data[i]['recipe']['instructions']:
                if(len(j['stepImages']) != 0):
                    imageSteps.append({
                        'recipeId' : data[i]['recipeId'],
                        'stepNumber' : j['stepNumber'],
                        'title' : j['stepTitle'],
                        'description' : j['stepText'],
                        'images' : j['stepImages']
                    })
                    img_data = requests.get(j['stepImages'][0]['url']).content
                    with open( "images/" + str(ctr) + '.jpg', 'wb') as handler:
                        handler.write(img_data)
                    ctr += 1

                    
                else:
                    noImageSteps.append({
                        'recipeId' : data[i]['recipeId'],
                        'stepNumber' : j['stepNumber'],
                        'title' : j['stepTitle'],
                        'description' : j['stepText']
                    })

        # escrever o ficheiro
        f = open(self.imagesFileName, 'w')
        f.write(json.dumps(imageSteps))

        f = open(self.noImagesFileName, 'w')
        f.write(json.dumps(noImageSteps))

        return
