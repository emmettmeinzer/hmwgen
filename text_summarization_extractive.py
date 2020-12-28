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

data = pd.read_csv('keyword_comment_cleaned.csv', nrows = 50)

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

num_sen = 5 #number of top ranked sentences
for i in range(num_sen):
    print(ranked_sentences[i][1])
    print("\n")

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of determiner, adjective, singular noun
  """
#PP: {<IN><NP>}               # Chunk prepositions followed by noun phrase
#VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
#CLAUSE: {<NP><VP>}           # Chunk noun phrase, verb phrase

rp = nltk.RegexpParser(grammar, loop=2)

store_parsed_pos_sen = []
tagged_POS = []
verbs = []
temp = []
temp1 = []
for i in range(num_sen):
    pos_sentences = TextBlob(ranked_sentences[i][1])
    tag_sentences = pos_sentences.tags
    tagged_POS.append(tag_sentences)
    for j in range(len(tag_sentences)):
        if (tag_sentences[j-2][1] == 'VB' or tag_sentences[j-2][1] == 'VBD' or tag_sentences[j-2][1] == 'VBG' or tag_sentences[j-2][1] == 'VBN' or tag_sentences[j-2][1] == 'VBP' or tag_sentences[j-2][1] == 'VBZ') and (tag_sentences[j-1][1] == 'TO') and (tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ'):
            if len(tag_sentences[j][0]) > 1:
                hold = tag_sentences[j-2][0] + ' ' + tag_sentences[j-1][0] + ' ' + tag_sentences[j][0]
                temp.append(hold)
        elif (tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ') and (tag_sentences[j-1][1] == 'MD'):
            if len(tag_sentences[j][0]) > 1:
                hold = tag_sentences[j-1][0] + ' ' + tag_sentences[j][0]
                temp.append(hold)
        elif tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ':
            if len(tag_sentences[j][0]) > 1:
                temp.append(tag_sentences[j][0].lemmatize("v"))
        if (tag_sentences[j][1] == 'RB' or tag_sentences[j][1] == 'RBR' or tag_sentences[j][1] == 'RBS' or tag_sentences[j][1] == 'JJ' or tag_sentences[j][1] == 'JJR' or tag_sentences[j][1] == 'JJS' or tag_sentences[j][1] == 'DT'):
            if len(tag_sentences[j][0]) > 1:
                if (j+2) < len(tag_sentences):
                    hold1 = tag_sentences[j][0] + ' ' + tag_sentences[j+1][0] + ' ' + tag_sentences[j+2][0]
                    temp1.append(hold1)
                elif (j+1) < len(tag_sentences):
                    hold1 = tag_sentences[j][0] + ' ' + tag_sentences[j+1][0]
                    temp1.append(hold1)
                else:
                    hold1 = tag_sentences[j][0]
                    temp1.append(hold1)
    verbs.append(temp)
    #addition.append(temp1)
    temp = []
    temp1 = []
    parse_sentences = rp.parse(tag_sentences)
    #parse_sentences.draw()
    store_parsed_pos_sen.append(parse_sentences)
    
addition = []
sen_phrases = []
hold = []
i = 0
for j in range(num_sen):
    temp = tagged_POS[j]
    while (i < len(temp)):
        if temp[i][1] == 'DT' or temp[i][1] == 'PDT' or temp[i][1] == 'IN' or temp[i][1] == 'JJ' or temp[i][1] == 'JJR' or temp[i][1] == 'JJS':
            hold.append(temp[i][0])
            i += 1
            while(i < len(temp)-1):
                if (temp[i][1] == 'NN' or temp[i] == 'RB' or temp[i][1] == 'VBG' or temp[i][1] == 'DT' or temp[i][1] == 'PDT' or temp[i][1] == 'IN' or temp[i][1] == 'JJ' or temp[i][1] == 'JJR' or temp[i][1] == 'JJS'):
                    hold.append(temp[i][0])
                    i += 1
                elif (temp[i][1] == 'NN' or temp[i][1] == 'NNS' or temp[i][1] == 'NNP' or temp[i][1] == 'NNPS' or temp[i][1] == 'IN') and (temp[i+1][1] == 'NN' or temp[i+1] == 'RB' or temp[i+1][1] == 'VBG' or temp[i+1][1] == 'DT' or temp[i+1][1] == 'PDT' or temp[i+1][1] == 'IN' or temp[i+1][1] == 'JJ' or temp[i+1][1] == 'JJR' or temp[i+1][1] == 'JJS'):
                    hold.append(temp[i][0])
                    i += 1
                elif (temp[i][1] == 'NN' or temp[i][1] == 'NNS' or temp[i][1] == 'NNP' or temp[i][1] == 'NNPS') and (temp[i+1][1] != 'NN' or temp[i+1] != 'RB' or temp[i+1][1] != 'VBG' or temp[i+1][1] != 'DT' or temp[i+1][1] != 'PDT' or temp[i+1][1] != 'IN' or temp[i+1][1] != 'JJ' or temp[i+1][1] != 'JJR' or temp[i+1][1] != 'JJS'):
                    hold.append(temp[i][0])
                    i += 1
                    if len(hold) > 1:
                        hold = ' '.join(hold)
                        sen_phrases.append(hold)
                    hold = []
                    break
                else:
                    i += 1
                    if len(hold) > 1:
                        hold = ' '.join(hold)
                        sen_phrases.append(hold)
                    hold = []
        else:
            i += 1
    i = 0
    addition.append(sen_phrases)
    sen_phrases = []

noun_phrase = []
hold = []
for i in range(num_sen):
    for j in range(len(store_parsed_pos_sen[i])):
        if type(store_parsed_pos_sen[i][j]) == nltk.tree.Tree:
            hold.append(store_parsed_pos_sen[i][j])
        else:
            None
    noun_phrase.append(hold)
    hold = []

str_np = []
temp = []
for i in range(len(noun_phrase)):
    for j in range(len(noun_phrase[i])):
        temp.append(str(noun_phrase[i][j]))
    str_np.append(temp)
    temp = []

split_np = []
temp =[]
for i in range(len(str_np)):
    for j in range(len(str_np[i])):
        temp.append(str_np[i][j].split())
    split_np.append(temp)
    temp = []
    
noun_phrase = []
temp1 = []
temp2 = []
for i in range(len(split_np)):
    for j in range(len(split_np[i])):
        for k in range(len(split_np[i][j])-1):
            temp1.append(split_np[i][j][k+1])
        temp2.append(temp1)
        temp1 = []
    noun_phrase.append(temp2)
    temp2 = []
    
np_clean = []
temp1 = []
temp2 = []
for i in range(len(noun_phrase)):
    for j in range(len(noun_phrase[i])):
        for k in range(len(noun_phrase[i][j])):
            hold = noun_phrase[i][j][k].rsplit('/',1)
            temp1.append(hold[0])
        temp2.append(temp1)
        temp1 = []
    np_clean.append(temp2)
    temp2 = []
    
nouns = []
temp = []
phrase = ''
for i in range(len(np_clean)):
    for j in range(len(np_clean[i])):
        for k in range(len(np_clean[i][j])):
            if k == (len(np_clean[i][j])-1):
                    phrase += np_clean[i][j][k]
            else:
                    phrase += np_clean[i][j][k] + ' '
        if len(np_clean[i][j][k]) > 1:
            temp.append(phrase)
        else:
            None
        phrase = ''
    nouns.append(temp)
    temp = []
    
from textblob import TextBlob
from textblob import Word
from textblob.wordnet import VERB
 
for i in range(len(verbs)):
    #temp = TextBlob(ranked_sentences[i][1], analyzer = NaiveBayesAnalyzer())
    #sent_rate = temp.sentiment
    #if sent_rate[0] == 'neg': 
        for j in range(len(verbs[i])):
            text_word = Word(verbs[i][j])
            for synset in text_word.get_synsets(pos=VERB):
                for lemma in synset.lemmas():
                    if lemma.antonyms():
                        verbs[i].append(lemma.antonyms()[0].name())
    #else:
        temp = []
        for j in range(len(verbs[i])):
            text_word = Word(verbs[i][j])
            for synset in text_word.get_synsets(pos=VERB):
                extract = synset.name().rsplit('.',2)
                temp.append(extract[0])
                temp = list(set(temp))
        for q in range(len(temp)):
            verbs[i].append(temp[q])
        temp = []
        
for i in range(len(verbs)):
    for j in range(len(verbs[i])):
        temp = list(verbs[i][j-1])
        for k in range(len(temp)):
            if temp[k] == "'":
                verbs[i].pop(j-1)
            elif temp[k] == "_":
                temp[k] = " "
                temp1 = ''.join(temp)
                verbs[i][j-1] = temp1
            else:
                None
        temp = None
        
hold = []
for i in range(num_sen):
    for j in range(len(addition[i])):
        temp = list(addition[i][j])
        for k in range(len(temp)):
            if temp[k] == "'" or temp[k] == ".":
                None
            else:
                hold.append(temp[k]) 
        addition[i][j] = ''.join(hold)
        hold = []

HMW = []
for i in range(num_sen):
        for j in range(len(verbs[i])):
            for k in range(len(addition[i])):
                #hold = "How might we " + verbs[i][j] + ' ' + addition[i][k] + '?'
                hold = verbs[i][j] + ' ' + addition[i][k]
                HMW.append(hold)
                
"""
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent‘s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO to go ‘to‘ the store.
UH interjection errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
"""
