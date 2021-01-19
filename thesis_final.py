# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:38:57 2021

@author: Emmett
"""
import pandas as pd
import numpy as py

import re
import os
import math
import pickle

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)

import nltk
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

from textblob import TextBlob
from textblob import Word
from textblob.wordnet import VERB

## PROPER DIRECTORY ##
os.chdir(r"C:\Users\emmet\.spyder-py3-dev\REU_Project")

## POPULATE CORPUS OF REDDIT SAMPLES ##
samp = pd.read_csv('RedditSamples.csv', encoding='cp1252')
corpus = []
for i in samp['Reddit Samples']:
    corpus.append(i)

## FEED EACH SAMPLE THROUGH T5 SUMMARIZER ##

#max_length/min_length: min/max number of tokens in summary
#length_penalty: penalizes model for producing summary above/below the maximum and minimum length thresholds
#num_beams: numbers of beams that explore potential token for most promising predictions
loop_len = len(corpus)
summaries = []
for i in range(loop_len):
    inputs = tokenizer.encode("summarize: " + corpus[i], return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=25, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
    summary = summary.split(' ', 1)[1]
    summaries.append(summary)
    inputs = None
    summary_ids = None
    summary = None
    
## WRITE SUMMARIES TO CSV FILE ##
write_to_csv = []
hold = []
for i in range(len(summaries)):
    hold.append(summaries[i])
    hold.append(corpus[i])
    write_to_csv.append(hold)
    hold = []
    
summary_df = pd.DataFrame(write_to_csv, columns=['T5 Summary', 'Original Statement'])
summary_df.to_csv("T5 Summaries.csv")

## EXTRACT KEY VERBS AND NOUN PHRASES FROM SUMMARIES USING TEXTBLOB ##
ranked_sentences = summaries
num_sen = len(ranked_sentences)

grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of determiner, adjective, singular noun
  """
rp = nltk.RegexpParser(grammar, loop=2)

store_parsed_pos_sen = []
tagged_POS = []
verbs = []
temp = []
temp1 = []
for i in range(num_sen):
    pos_sentences = TextBlob(ranked_sentences[i])
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

## CLEAN NOUN PHRASES ##
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
    
## GENERATE APPLICABLE SYNONYMS AND ANTONYMS OF KEY VERBS FROM SUMMARIES ##
for i in range(len(verbs)):
    for j in range(len(verbs[i])):
        text_word = Word(verbs[i][j])
        for synset in text_word.get_synsets(pos=VERB):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    verbs[i].append(lemma.antonyms()[0].name())
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
        if j > len(verbs[i]):
            break
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

## GENERATE HOW MIGHT WE QUESTION STEMS
HMW = []
for i in range(num_sen):
        for j in range(len(verbs[i])):
            for k in range(len(addition[i])):
                #hold = "How might we " + verbs[i][j] + ' ' + addition[i][k] + '?'
                hold = verbs[i][j] + ' ' + addition[i][k]
                HMW.append(hold)

## NONSENSE DETECTOR ##                
#take in 10,000 most common English words
accepted_words = [line.rstrip() for line in open('wordlist.10000.txt')]
#enumerate each word from 1 to 10,000
pos = dict([(char, idx) for idx, char in enumerate(accepted_words)])

#split sentence into array of individual words
#remove non-alphabetic characters
#make sentence lowercase
def normalize(line):
    norm = []
    for word in line.split():
        word = re.sub('[^A-Za-z0-9]+', '', str(word))
        if word.lower() in accepted_words:
            norm.append(word.lower())
        else:
            None
    return norm

#return all n grams from line after normalizing
def ngram(n, l):
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield filtered[start:start + n]
        
def avg_transition_prob(l, log_prob_mat):
    #Return the average transition probability from l through log_prob_mat
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]] #find the probability distribution of pos[a] occuring next to pos[b]
        transition_ct += 1
    #base e exponention to convert from log scale probability to traditonal scale
    return math.exp(log_prob / (transition_ct or 1))

k = len(accepted_words)
# Assume we have seen 10 of each character pair.  This acts as a kind of
# prior or smoothing factor.  This way, if we see a character transition
# live that we've never observed in the past, we won't assume the entire
# string has 0 probability.
counts = [[10 for i in range(k)] for i in range(k)]

# Count transitions from a baseline text with relevant vocabulary
baseline_df = pd.read_csv('keyword_comment_cleaned.csv')
for line in baseline_df['comment_body']:
    for a, b in ngram(2, line):
        counts[pos[a]][pos[b]] += 1

# Normalize the counts so that they become log probabilities.  
# We use log probabilities rather than straight probabilities to avoid
# numeric underflow issues with long texts.
# This contains a justification:
# http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
for i, row in enumerate(counts):
    s = float(sum(row))
    for j in range(len(row)):
        row[j] = math.log(row[j] / s) #log of element divided by sum of the row

# And pick a threshold halfway between the worst good and best bad HMW questions.
threshold = 0.005
sensical_questions = []
for i in range(len(HMW)):
    if ((avg_transition_prob(HMW[i], counts) > threshold) == True):
        sensical_questions.append(HMW[i])
    else:
        None

for i in range(len(sensical_questions)):
    sensical_questions[i] = ("How might we " + sensical_questions[i] + "?")        
questions_df = pd.DataFrame(sensical_questions, columns=['How Might We Questions'])
questions_df.to_csv("HMW Questions.csv")