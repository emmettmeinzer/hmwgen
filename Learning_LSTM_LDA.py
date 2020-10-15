# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:50:55 2020

@author: Emmett
"""

import json
import nltk
#import lda2
import string
import copy
import pandas as pd
import numpy as np
import keras.backend as K

from keras import regularizers
from keras.models import Model
from numpy import linalg as LA
from nltk.corpus import stopwords
from scipy.special import gammaln
from keras.models import Sequential
from scipy.sparse import csr_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense, Activation, Embedding, LSTM

def get_x_lstm(max_vocab, vocab):
    tokenizer = Tokenizer(nb_words = max_vocab, lower=True, split=' ')
    tokenizer.fit_on_texts(vocab)
    vocab_seq = tokenizer.texts_to_sequences(vocab)
    return pad_sequences(vocab_seq)

def sampleFromDirichlet(samp):
    return np.random.dirichlet(samp)

def sampleFromCategorical(samp):
    samp = np.exp(samp)/np.exp(samp).sum()
    return np.random.multinomial(1, samp).argmax()

def word_indices(wordOccuranceVec):
    for i in wordOccuranceVec.nonzero()[0]:
        for j in range(int(wordOccuranceVec[i])):
            yield i

#maximum number of features
MAX_VOCAB_SIZE = 100
def processCorpus(samples, window_size = 5, MAX_VOCAB_SIZE = MAX_VOCAB_SIZE):
    vectorizer = CountVectorizer(analyzer="word", tokenizer = None)
    train_data_features = vectorizer.fit_transform(samples)
    words = vectorizer.get_feature_names()
    vocab = dict(zip(words, np.arange(len(words))))
    inv_vocab = dict(zip(np.arange(len(words)), words))
    wordOccuranceMatrix = train_data_features.toarray()
    return wordOccuranceMatrix, vocab, words

sort = True

import gensim
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text

corpus = []
sampNum = 1
while (sampNum < 186):
    fileOpen = open("sample"+str(sampNum)+".txt","r")
    temp = fileOpen.readlines()
    temp1 = strip_non_alphanum(str(temp))
    temp2 = strip_punctuation(temp1)
    final = strip_multiple_whitespaces(temp2)
    #final = stem_text(temp3)
    corpus.append(final)
    sampNum += 1

stoplist = set('a about above after again against all am an and any are arent\
               as also at be because been before being below between both but\
               by cant cannot could couldnt did didnt do does doesnt doing dont\
               down during each els few for from further had hadnt has have havent\
               having he hed hes her here heres hers herself him himself his\
               how hows i id ill im ive if in into is isnt it its itself lets\
               me more most mustnt my myself no nor not of off on once only or\
               other ought our ours ourselves out over own same shant she shes\
               should shouldnt so some such than that thats the their theirs\
               them themselves then there theres these they theyd theyll theyre\
               theyve this those through to too under until up very was wasnt\
               we wed were weve were werent what whats when whens which while\
               who whos whom why whys with wont would wouldnt you youd youll\
               youre youve your yours yourself yourselves ll ve s ar mayb ha re\
               us thi isn a b c d e f g h i j k l m n o p q r s t u v w x y z\
               hi will can get back go don wa let atc ok ani mi thei whenev make\
               just take aw know sai good baltimor jetblu lol thank thanks like\
               vari might less highest billion nice probabl lot fuck shit sure\
               feel dure befor realli work veri chanc see awai onc onli dy aren\
               100 someth thing even happen becaus wai everi much help want think\
               fear flight plane fly mai time dai\
               1 2 3 4 5 6 7 8 9 10'.split())

corpusList = [i for item in corpus for i in item.split()]
matrix, vocabulary, words = processCorpus(corpusList)

#standardize text -- makes all characters lowercase and removes common words
texts = [[word for word in document.lower().split() if word not in stoplist]
        for document in corpus]

#count number of times that word appears in corpus
#pair frequency with respective word in new array
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
        
corpus_removeOne = [[token for token in text if frequency[token]>1] for text in texts]

from gensim import corpora
#add corpora to dictionary
dictionary = corpora.Dictionary(corpus_removeOne)
#save dictionary for future reference
dictionary.save('C:\\Users\\emmet\\.spyder-py3-dev\\redditTest.dict') #location of document in computer
#dict = gensim.corpora.Dictionary.load('redditTest.dict')

#assign numeric id to each token in dictionary
dictID = dictionary.token2id

#converts each word into vector following same process as example
bow_corpus = [dictionary.doc2bow(text) for text in corpus_removeOne]
corpora.MmCorpus.serialize('redditTest.mm', bow_corpus)
corp = gensim.corpora.MmCorpus('redditTest.mm')

print(bow_corpus)

from gensim import models
#from gensim.models import TfidfModel
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
for doc in corpus_tfidf:
    if doc == []:
        None
    else:
        print(doc)

lda = models.LdaModel(bow_corpus, id2word=dictionary, num_topics=9)
corpus_LDA = lda[bow_corpus]

print(corpus_LDA)

"Long Short Term Memory"

lstm_out = 128
batch_size = 8
p_embedding_lstm = 200

X = get_x_lstm(MAX_VOCAB_SIZE, corpusList)

#Sequential: linear stack of layers
#Embedding: turns positive integers (indexes) into dense vectors of fixed size
#LSTM: long short term memory
#Dropout: regularization method where input and recurrent connections to LSTM
#units are probabilistically excluded from activation and weight updates while
#training a network
#Dense: densely-connected neural network layer
#Activation: element-wise activation function passed as the 'activation' argument
#Kernel: weights matrix created by the layer
#Compile: compile source into code object that can be executed by exec() or eval()

model = Sequential()
model.add(Embedding(MAX_VOCAB_SIZE, p_embedding_lstm, input_length = X.shape[0]))
model.add(LSTM(lstm_out, dropout = 0.2))
model.add(Dense(5, activation = 'tanh', name = "doc_latent_vector", kernel_regularizer = regularizers.l2()))
model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', metrics = ['accuracy'])
model.summary()

#Model: groups layers into an object with training and inference features
#Inputs: inputs of the model (object or list of objects)
#Outputs: outputs of the model

def get_last_layer_op():
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('doc_latent_vector').output)
    return intermediate_layer_model.predict(X)

"LOSS"



#PMF??
#HFT??
#LSTM
