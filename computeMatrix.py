import pickle
import numpy as np

class ComputeMatrix():
    def __init__(self, imageEmbeddingsfile, noImageEmbeddingsfile):
        self.imageEmbeddingsfile = imageEmbeddingsfile
        self.noImageEmbeddingsfile = noImageEmbeddingsfile


    def createMatrix(self):
        with open(self.imageEmbeddingsfile + '.pickle', "rb") as read_file:
            imageEmbeddings=pickle.load(read_file)
        with open(self.noImageEmbeddingsfile + '.pickle', "rb") as read_file:
            noImageEmbeddings=pickle.load(read_file)
        matrix = np.dot(noImageEmbeddings, np.transpose(imageEmbeddings))
        return matrix