import json as json


class CreateRecipesMap():
    def __init__(self, inputFileName, outputFileName):
        self.inputFileName = inputFileName
        self.outputFileName = outputFileName

    def createMap(self):
        with open(self.inputFileName, "r") as read_file:
            data = json.load(read_file)



        recipes = {}
        for i in data:
            for j in i['results']:
                recipes[j["_id"]] = j['_source']




        #escrever o ficheiro 
        f = open(self.outputFileName, 'w')
        f.write(json.dumps(recipes))

        return