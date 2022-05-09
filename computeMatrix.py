import pickle
import numpy as np

class ComputeMatrix():
    def __init__(self, imageEmbeddingsfile, stepEmbeddingsfile):
        self.imageEmbeddingsfile = imageEmbeddingsfile
        self.stepEmbeddingsfile = stepEmbeddingsfile


    def createMatrix(self):
        with open(self.imageEmbeddingsfile + '.pickle', "rb") as read_file:
            imageEmbeddings=pickle.load(read_file)
        with open(self.stepEmbeddingsfile + '.pickle', "rb") as read_file:
            stepEmbeddings=pickle.load(read_file)


        matrix = np.matmul(stepEmbeddings, np.transpose(imageEmbeddings))
        return matrix