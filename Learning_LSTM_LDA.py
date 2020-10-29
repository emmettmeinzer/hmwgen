# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:50:55 2020

@author: Emmett
"""

import nltk
nltk.download('wordnet')
import LDA_Sampler
import string
import copy
import pandas as pd
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

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

make_singularRoot = nltk.stem.WordNetLemmatizer()
remove_ws = nltk.tokenize.WhitespaceTokenizer()
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
               hi will can get [deleted]\
               1 2 3 4 5 6 7 8 9 10'.split())

def preprocess(pd):
    pd = pd.str.lower()
    pd = pd.str.replace('[{}]'.format(string.punctuation), ' ')
    pd = pd.apply(lambda x: [make_singularRoot.lemmatize(w) for w in remove_ws.tokenize(x)])
    pd = pd.apply(lambda x: [item for item in x if item not in stoplist])
    return pd.str.join(' ')

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
def processComments(samples, window_size = 5, MAX_VOCAB_SIZE = MAX_VOCAB_SIZE):
    #Convert the collection of comments to a matrix of token counts
    vectorizer = CountVectorizer(analyzer="word", tokenizer = None)
    #Learn the vocabulary dictionary and return term-document matrix
    train_data_features = vectorizer.fit_transform(samples)
    #Array mapping from feature integer indices to feature name
    words = vectorizer.get_feature_names()
    vocab = dict(zip(words, np.arange(len(words))))
    inv_vocab = dict(zip(np.arange(len(words)), words))
    wordOccuranceMatrix = train_data_features.toarray()
    return wordOccuranceMatrix, vocab, words

sort = True

import gensim
"""
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
"""
    
data = pd.read_csv('keyword_comment_cleaned.csv')

user_ids = list(data.comment_parent_id.unique())
post_ids = list(data.post_id.unique())

num_user_ids = len(user_ids)
num_post_ids = len(post_ids)

comment_score = np.zeros((num_user_ids, num_post_ids))

for idx, i in enumerate(data[["comment_parent_id", "comment_score", "post_id"]].values):
    comment_score[user_ids.index(i[0])][post_ids.index(i[2])] = i[1]
        
comment_score_normalized = comment_score/max(data.comment_score)

comments = [""] * num_post_ids

for i in data[["post_id", "comment_body"]].values:
    comments[post_ids.index(i[0])] += i[1]
comments = pd.DataFrame(comments)
comments = preprocess(comments[0])
comments.shape

#corpusList = [i for item in corpus for i in item.split()]
matrix, vocabulary, words = processComments(comments)
num_topics = 9
lambda_param = 0.8

#Probabilistic Matrix Factorization (PMF)
#From paper: "effective recommendation model that uses matrix factorization (MF) 
#technique to find the latent features of users and items from a probabilistic perspective"
#create user latent vector
user_weights = np.random.rand(num_topics, num_user_ids)
#create item (in this case, posts) latent vector
post_weights = np.random.rand(num_topics, num_post_ids)

beta = 0.01
alpha = 10*np.ones(num_topics)/num_topics

##############################################################################
#standardize text -- makes all characters lowercase and removes common words
texts = [[word for word in document.lower().split() if word not in stoplist]
        for document in comments]

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

lda_model = models.LdaModel(bow_corpus, id2word=dictionary, num_topics=9)
corpus_LDA = lda_model[bow_corpus]

print(corpus_LDA)
##############################################################################

lda = LDA_Sampler.LdaSampler(n_topics=num_topics, matrix_shape=matrix.shape, lambda_param=lambda_param)

"Long Short Term Memory"
lstm_out = 128
batch_size = 8
p_embedding_lstm = 200

X = get_x_lstm(MAX_VOCAB_SIZE, comments.values)

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
model.add(Embedding(MAX_VOCAB_SIZE, p_embedding_lstm, input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout = 0.2))
model.add(Dense(num_topics, activation = 'tanh', name = "doc_latent_vector", kernel_regularizer = regularizers.l2()))
model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', metrics = ['accuracy'])
model.summary()

#Model: groups layers into an object with training and inference features
#Inputs: inputs of the model (object or list of objects)
#Outputs: outputs of the model

def get_last_layer_op():
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('doc_latent_vector').output)
    return intermediate_layer_model.predict(X)

#PMF Loss function
#L = summation(Iij*(Rij - Rij_hat)^2)
#Regularization parameter: Iij = 1
#Normalized post rating: Rij
#Predicted post rating: Rij_hat; dot product of user latent vector and post latent vector (Rij_hat = u_i*v_i)
def get_l1():
    l1 = 0
    for i in range(num_user_ids):
        for j in range(num_post_ids):
            if comment_score_normalized[i][j] != 0:
                l1 += (comment_score_normalized[i][j] - np.dot(user_weights.T[i], post_weights.T[j]))**2
    return l1

#matrix normalization -- square root of the sum of the absolute squares of its elements
def get_l3():
    return LA.norm(user_weights, 'fro')

#matrix normalization -- square root of the sum of the absolute squares of its elements
def get_l4():
    return LA.norm(post_weights.T - get_last_layer_op(), 'fro')

#set regularization parameters
l3_coeff = l4_coeff = 0.01
#From paper: loss function is the summation of these three individual loss calculations
def get_total_loss():
    return (get_l1() + l3_coeff*get_l3() + l4_coeff*get_l4())

#Optimization

#topic parameters: k, njk, Nj
#njk: the number of times when topic k occurs in the document of item j
#Nj:  the total number of words in document
#k = peakiness: control the transformation between item vector v and topic distribution θ
#diff_lv: partial derivative of Loss function wrt rating-based latent vector v
def gradient_V(lda, lstm_last_layer):
    peakiness = 1
    lambda_t = 0.01
    param_Nj = matrix.sum(axis=1)
    param_njk = lda.nmz.copy()
    dt_distribution = lda.theta()
    #refer to equation 7 in the paper
    diff_lv = []
    for j in range(num_post_ids):
        temp_sums = [0]*num_topics
        for i in range(num_user_ids):
            if comment_score_normalized[i][j] != 0:
                temp_sums += (comment_score_normalized[i][j] - np.dot(user_weights.T[i], post_weights.T[j]))*user_weights.T[i]
        temp_sums += 2*l4_coeff*(post_weights.T[j] - lstm_last_layer[j])
        temp_sums -= lambda_t*peakiness*(param_njk[j] - param_Nj[j]*dt_distribution[j].sum())
        diff_lv.append(list(temp_sums))
    diff_lv = np.array(diff_lv)
    return diff_lv

#diff_lu: partial derivative of Loss function wrt user latent vector u
#Refer to equation 9 in the paper
def gradient_U():
    diff_lu = []
    for i in range(num_user_ids):
        temp_sums = [0]*num_topics
        for j in range(num_post_ids):
            if comment_score_normalized[i][j] != 0:
                temp_sums += (comment_score_normalized[i][j] - np.dot(user_weights.T[i], post_weights.T[j]))*post_weights.T[j]
        temp_sums += 2*l3_coeff*user_weights.T[i]
        diff_lu.append(list(temp_sums))
    diff_lu = np.array(diff_lu)
    return diff_lu

#phi: word distribution
#nkw: the number of times that word w occurs in topic k
#Nw is the word vocabulary size of the document corpus
#Nk is the number of words in topic k; nj,k is the number of times when topic k occurs in the document of item j
#zw: corresponding normalizers defined at end of Section 4
#diff_phi: partial derivative of Loss function wrt word distribution phi
#Refer to equation 10 in the paper
def gradient_Phi(lda, phi_weights):
    param_nkw = lda.nzw.T
    param_Nk = lda.nzw.sum(axis=1)
    diff_phi = []
    for i in range(MAX_VOCAB_SIZE):
        param_zw = np.exp(phi_weights[i].sum())
        temp_phi = []
        for j in range (num_topics):
            temp_phi.append(param_nkw[i, j] - (param_Nk[j]*np.exp(phi_weights[i, j])/param_zw))
        diff_phi.append(temp_phi)
    diff_phi = np.array(diff_phi)
    return diff_phi

maxiter_hft = 10
learning_rate_pmf = learning_rate_hft = 0.01
phi_weights = np.random.rand(MAX_VOCAB_SIZE, num_topics)

#Paper gives 30 iterations as optimum number of cycles
#It takes far too long for my computer to run
#After 3 iterations, loss trends towards infinity
iters = 3
for i in range(iters):
    lda.run(matrix, maxiter_hft)
    temp = num_topics
    for i in range(temp):
        #tries to fit dataset X
        model.fit(X, post_weights.T, epochs = num_topics, batch_size = 128)
        lstm_last_layer = get_last_layer_op()
        
        print("\nExtracting Gradients...")
        gradient_v = gradient_V(lda, lstm_last_layer)
        gradient_u = gradient_U()
        gradient_phi = gradient_Phi(lda, phi_weights)
        
        print("\nUpdating Gradients...")
        user_weights -= learning_rate_pmf * gradient_u.T
        post_weights -= learning_rate_pmf * gradient_v.T
        phi_weights -= learning_rate_hft * gradient_phi
        
        print(get_l1(), get_l3(), get_l4())


'Create a Data Set from which the Model will make its Prediction'        
data_predict = pd.read_csv('test_input.csv')

post_ids_predict = list(data_predict.post_id.unique())
num_post_ids_predict = len(post_ids_predict)

comments_predict = [""] * num_post_ids_predict

for i in data_predict[["post_id", "comment_body"]].values:
    comments_predict[post_ids_predict.index(i[0])] += i[1]
comments_predict = pd.DataFrame(comments_predict)
comments_predict = preprocess(comments_predict[0])
comments_predict.shape

test_input = get_x_lstm(MAX_VOCAB_SIZE, comments_predict.values)
test_output = model.predict(test_input, verbose=0)
print(test_output)
