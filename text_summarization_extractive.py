# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:34:00 2020

@author: Emmett
"""
#EXTRACTIVE SUMMARIZATION

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import re

from nltk.corpus import stopwords
stoplist = stopwords.words('english')

def remove_stoplist(sentences):
    new_sentence = " ".join([word for word in sentences if word not in stoplist])
    return new_sentence

def preprocess(sentences):
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [letter.lower() for letter in clean_sentences]
    clean_sentences = [remove_stoplist(word.split()) for word in clean_sentences]
    return clean_sentences

data = pd.read_csv('keyword_comment_cleaned.csv')

sentences = []
for i in data['comment_body']:
    sentences.append(sent_tokenize(i))
    
sentences = [word for i in sentences for word in i]

clean_sentences = preprocess(sentences)

word_embeddings = {}
file = open('glove.6B.100d.txt', encoding='utf-8')
for line in file:
    values = line.split()
    word = values[0]
    coeffs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coeffs
file.close()

sentence_vectors = []
for i in clean_sentences:
    #if the number of characters in the sentence is not zero
    if len(i) != 0:
        #fetch vectors (size 100) for the constituent words in a sentence
        #take average of those vectors to get a consolidated vector for the sentence
        vec = sum([word_embeddings.get(word, np.zeros(100)) for word in i.split()])/(len(i.split())+0.001)
    else:
        #if there is no sentence in that element of clean_sentences, vector is zeros
        vec = np.zeros(100)
    sentence_vectors.append(vec)
    
#initialize NxN similarity matrix (N = number of sentences)
sim_matrix = np.zeros([len(sentences), len(sentences)])

for i in range(len(sentences)):
    for j in range(len(sentences)):
        #parce through non-diagonal elements of similarity matrix
        if i != j:
            #cosine similarity: cosine of angle between two n-dimensional vectors (dot product of two vectors divided by product of both vector's magnitudes)
            sim_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

nx_graph = nx.from_numpy_array(sim_matrix)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i], sen) for i, sen in enumerate(sentences)), reverse=True)

num_sen = 10 #number of top ranked sentences
for i in range(num_sen):
    print(ranked_sentences[i][1])