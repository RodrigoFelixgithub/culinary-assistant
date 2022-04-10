from transformers import AutoTokenizer, AutoModel
import pprint as pp
from pathlib import Path
import search as s
import torch
import torch.nn.functional as F
import spacy as spacy
from spacy import displacy
import sys
import json as json
import spacy
import math
import numpy
import time



class SearchJson():
    def __init__(self, client, index_name, tokenizer, model):
        self.client = client
        self.index_name = index_name
        self.tokenizer = tokenizer
        self.model = model

    def precision_func(self, no_false_positives, no_true_positives):
        return no_true_positives/(no_false_positives + no_true_positives)
    
    def recall_func(self, no_false_negatives, no_true_positives):
        return no_true_positives/(no_false_negatives + no_true_positives)
    
    def f1_scoreFunc(self, precision, recall):
        return 2*precision*recall/(precision+recall)

    def list_copy(self, list):
        aux = []
        for i in list:
            aux.append(i)
        return aux

    def searchJson(self, jsonFile):
        positive_annotations = {}
        nlp = spacy.load("en_core_web_sm")
        with open(jsonFile, "r") as read_file:
            annots = json.load(read_file)
        for query in annots.keys():
            positive_annotations[query] = []
            for i in annots[query]:
                if int(i['label']) > 0:
                    recipeTitleDoc = nlp(i['answer'])
                    recipeTitleString = ' '.join([token.lemma_ for token in recipeTitleDoc if not token.is_stop and token.is_alpha])
                    positive_annotations[query].append(recipeTitleString)
                
        no_true_positives = 0
        no_false_positives = 0
        no_false_negatives = 0
        
        qmaker = s.Search(self.client, self.index_name, self.tokenizer, self.model)
        startTime = time.time()
        for query in annots.keys():
            nresultsaux = len(positive_annotations[query])
            res = qmaker.queryOpenSearch(query, nresultsaux, None, None, None,None, 1000)
            posit_annotation_copy = self.list_copy(positive_annotations[query])
            no_true_positives_aux = 0
            no_false_positives_aux = 0
            for i in res['hits']['hits']:    
                t = i['fields']['title'][0]
                if(t in positive_annotations[query]):
                    no_true_positives_aux += 1
                    if t in posit_annotation_copy:
                        posit_annotation_copy.remove(t)
                else:
                    no_false_positives_aux += 1
            no_false_negatives += len(posit_annotation_copy)       
            no_true_positives += no_true_positives_aux
            no_false_positives += no_false_positives_aux
        endTime = time.time()
        precision_score = self.precision_func(no_false_positives, no_true_positives)
        recall_score = self.recall_func(no_false_negatives, no_true_positives)
        f1_score = self.f1_scoreFunc(precision_score, recall_score)
        print("Precision: " + str(precision_score) +"\nRecall score: " + str(recall_score) + "\nF1 Score: "+ str(f1_score) + "\nTime took: "+ str(endTime - startTime))