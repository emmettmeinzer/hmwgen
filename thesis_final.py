# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:38:57 2021

@author: Emmett
"""
import pandas as pd
import numpy as np
from numpy import linalg

import re
import os
import csv
import math
import time
import pickle
import random

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained('t5-11b')
model = AutoModelWithLMHead.from_pretrained('t5-11b', return_dict=True)

import nltk
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Word
from textblob.wordnet import VERB

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from datetime import datetime

stoplist = ['b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def remove_stoplist(sentences):
    new_sentence = " ".join([word for word in sentences if word not in stoplist])
    return new_sentence

def preprocess(corpus):
    clean_sentences = []
    for i in range(len(corpus)):
        new_sentences = pd.Series(corpus[i]).str.replace("[^a-zA-Z]", " ")
        new_sentences = [letter.lower() for letter in new_sentences]
        clean_sentences.append(" ".join([word for word in new_sentences[0].split()]))
        clean_sentences = [remove_stoplist(word.split()) for word in clean_sentences]
    return clean_sentences

samp = pd.read_csv('RedditSamples.csv', encoding='cp1252')
corpus = []
for i in samp['Reddit Samples']:
    corpus.append(i)
clean_samples = preprocess(corpus)

baseline_df = pd.read_csv('keyword_comment_cleaned.csv')
baseline_corpus = []
for i in baseline_df['comment_body']:
    baseline_corpus.append(i)
clean_baseline_corpus = preprocess(baseline_corpus)
baseline_df['cleaned'] = clean_baseline_corpus

loop_len = len(clean_samples)
summaries = []
for i in range(loop_len):
    inputs = tokenizer.encode("summarize: " + clean_samples[i], return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=25, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
    summary = summary.split(' ', 1)[1]
    summaries.append(summary)
    inputs = None
    summary_ids = None
    summary = None
    
write_to_csv = []
hold = []
for i in range(len(summaries)):
    hold.append(summaries[i])
    hold.append(corpus[i])
    write_to_csv.append(hold)
    hold = []
    
summary_df = pd.DataFrame(write_to_csv, columns=['T5 Summary', 'Original Statement'])
summary_df.to_csv("T5 Summaries.csv")
clean_summaries = preprocess(summaries)

def get_addition_verbs(sentences, num_sen):
    grammar = r"""
    NP: {<DT|JJ|NN.*>+}          # Chunk sequences of determiner, adjective, singular noun
    """
    rp = nltk.RegexpParser(grammar, loop=2)
    
    store_parsed_pos_sen = []
    tagged_POS = []
    verbs = []
    verbs_plus_one = []
    
    for i in range(num_sen):
        pos_sentences = TextBlob(sentences[i])
        tag_sentences = pos_sentences.tags
        tagged_POS.append(tag_sentences)
    
        temp = []
        plus_one = []
        for j in range(len(tag_sentences)):
            if j > 1:
                if (tag_sentences[j-2][1] == 'MD' or tag_sentences[j-2][1] == 'VB' or tag_sentences[j-2][1] == 'VBD' or tag_sentences[j-2][1] == 'VBG' or tag_sentences[j-2][1] == 'VBN' or tag_sentences[j-2][1] == 'VBP' or tag_sentences[j-2][1] == 'VBZ' or tag_sentences[j-2][1] == 'TO') and (tag_sentences[j-1][1] == 'MD' or tag_sentences[j-1][1] == 'VB' or tag_sentences[j-1][1] == 'VBD' or tag_sentences[j-1][1] == 'VBG' or tag_sentences[j-1][1] == 'VBN' or tag_sentences[j-1][1] == 'VBP' or tag_sentences[j-1][1] == 'VBZ' or tag_sentences[j-1][1] == 'TO') and (tag_sentences[j][1] == 'MD' or tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ' or tag_sentences[j][1] == 'TO'):
                    if (len(tag_sentences)-j) > 2:
                        hold = tag_sentences[j-2][0] + ' ' + tag_sentences[j-1][0] + ' ' + tag_sentences[j][0]
                        plus = hold + ' ' + tag_sentences[j+1][0]
                        plus_one.append(plus)
                        temp.append(hold)
                    else:
                        hold = tag_sentences[j-2][0] + ' ' + tag_sentences[j-1][0] + ' ' + tag_sentences[j][0]
                        temp.append(hold)
                elif (tag_sentences[j-1][1] == 'MD' or tag_sentences[j-1][1] == 'VB' or tag_sentences[j-1][1] == 'VBD' or tag_sentences[j-1][1] == 'VBG' or tag_sentences[j-1][1] == 'VBN' or tag_sentences[j-1][1] == 'VBP' or tag_sentences[j-1][1] == 'VBZ' or tag_sentences[j-1][1] == 'TO') and (tag_sentences[j][1] == 'MD' or tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ' or tag_sentences[j][1] == 'TO'):
                    if (len(tag_sentences)-j) > 2:
                        hold = tag_sentences[j-1][0] + ' ' + tag_sentences[j][0]
                        plus = hold + ' ' + tag_sentences[j+1][0]
                        plus_one.append(plus)
                        temp.append(hold)
                    else:
                        hold = tag_sentences[j-1][0] + ' ' + tag_sentences[j][0]
                        temp.append(hold)
                else:
                    if (tag_sentences[j][1] == 'MD' or tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ' or tag_sentences[j][1] == 'TO'):
                        if (len(tag_sentences)-j) > 2:
                            plus = tag_sentences[j][0] + ' ' + tag_sentences[j+1][0]
                            plus_one.append(plus)
                            temp.append(tag_sentences[j][0].lemmatize("v"))
                        else:
                            temp.append(tag_sentences[j][0].lemmatize("v"))
            elif j > 0:
                if (tag_sentences[j-1][1] == 'MD' or tag_sentences[j-1][1] == 'VB' or tag_sentences[j-1][1] == 'VBD' or tag_sentences[j-1][1] == 'VBG' or tag_sentences[j-1][1] == 'VBN' or tag_sentences[j-1][1] == 'VBP' or tag_sentences[j-1][1] == 'VBZ' or tag_sentences[j-1][1] == 'TO') and (tag_sentences[j][1] == 'MD' or tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ' or tag_sentences[j][1] == 'TO'):
                    if (len(tag_sentences)-j) > 2:
                        hold = tag_sentences[j-1][0] + ' ' + tag_sentences[j][0]
                        plus = hold + ' ' + tag_sentences[j+1][0]
                        plus_one.append(plus)
                        temp.append(hold)
                    else:
                        hold = tag_sentences[j-1][0] + ' ' + tag_sentences[j][0]
                        temp.append(hold)
                else:
                   if (tag_sentences[j][1] == 'MD' or tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ' or tag_sentences[j][1] == 'TO'):
                       if (len(tag_sentences)-j) > 2:
                           plus = tag_sentences[j][0] + ' ' + tag_sentences[j+1][0]
                           plus_one.append(plus)
                           temp.append(tag_sentences[j][0].lemmatize("v"))
                       else:
                           temp.append(tag_sentences[j][0].lemmatize("v"))                       
            else:
                if (tag_sentences[j][1] == 'MD' or tag_sentences[j][1] == 'VB' or tag_sentences[j][1] == 'VBD' or tag_sentences[j][1] == 'VBG' or tag_sentences[j][1] == 'VBN' or tag_sentences[j][1] == 'VBP' or tag_sentences[j][1] == 'VBZ' or tag_sentences[j][1] == 'TO'):
                    if (len(tag_sentences)-j) > 2:
                        plus = tag_sentences[j][0] + ' ' + tag_sentences[j+1][0]
                        plus_one.append(plus)
                        temp.append(tag_sentences[j][0].lemmatize("v"))
                    else:
                        temp.append(tag_sentences[j][0].lemmatize("v"))
        
        verbs.append(temp)
        verbs_plus_one.append(plus_one)
        temp = []
        plus_one = []
        parse_sentences = rp.parse(tag_sentences)
        #parse_sentences.draw()
        store_parsed_pos_sen.append(parse_sentences)
    
    addition = []
    sen_phrases = []
    hold = []
    i = 0
    j = 0
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
        
    final_verbs = []
    final_verbs_plus_one = []
    hold = []
    temp = []
    for i in range(len(verbs)):
        for j in verbs[i]:
            if j not in final_verbs:
                hold.append(j)
        for k in verbs_plus_one[i]:
            if k not in final_verbs_plus_one:
                temp.append(k)
        final_verbs.append(hold)
        final_verbs_plus_one.append(temp)
        hold = []
        temp = []
        
    return addition, final_verbs, final_verbs_plus_one, tagged_POS

addition, verbs, verbs_plus_one, summary_POS = get_addition_verbs(clean_summaries, len(clean_summaries))
baseline_addition, baseline_verbs, baseline_verbs_plus_one, baseline_POS = get_addition_verbs(clean_baseline_corpus, len(clean_baseline_corpus))
num_sen = len(clean_summaries)

num_sen = len(clean_summaries)
HMW = []
hold = []
for i in range(num_sen):
        for j in range(len(verbs[i])):
            for k in range(len(addition[i])):
                hmw_sen = verbs[i][j] + ' ' + addition[i][k]
                hold.append(hmw_sen)
        HMW.append(hold)
        hold = []
        
summary_tag = []
temp = []
hold = []
for i in range(len(HMW)):
    for j in range(len(HMW[i])):
        hold.append(HMW[i][j])
        hold.append(corpus[i])
        temp.append(hold)
        hold = []
    summary_tag.append(temp)
    temp = []

def get_acceptable_words(corpus):
    all_words = []
    for i in range(len(corpus)):
        all_words.append(corpus[i].split())
    
    all_words = [i for sub in all_words for i in sub]
    
    acceptable_words = []
    for i in all_words:
        if not i in acceptable_words:
            acceptable_words.append(i)
            
    return sorted(acceptable_words)

accepted_words = get_acceptable_words(clean_baseline_corpus)
pos = dict([(char, index) for index, char in enumerate(accepted_words)])

def normalize(line):
    norm = []
    for word in line.split():
        word = re.sub('[^A-Za-z0-9]+', '', str(word))
        if word.lower() in accepted_words:
            norm.append(word.lower())
        else:
            None
    return norm

def ngram(n, l):
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield filtered[start:start + n]

def pos_ngram(n, l):
    for start in range(0, len(l) - n + 1):
        yield l[start:start + n]

def revised_ngram(n, l):
    ngrams = []
    for start in range(0, len(l) - n + 1):
        ngrams.append(l[start:start + n])
    
    return ngrams

def avg_word_prob(l, log_prob_mat, pos):
    log_prob = 0.0
    count = 0
    for a, b in ngram(2, l):
        if a not in pos or b not in pos:
            log_prob += 0
            count += 1
        else:
            log_prob += log_prob_mat[pos[a]][pos[b]]
            count += 1

    return math.exp(log_prob / (count or 1))

def avg_pos_prob(l, log_prob_mat, pos):
    log_prob = 0.0
    count = 0
    for a, b in pos_ngram(2, l):
        if a not in pos or b not in pos:
            log_prob += 0
            count += 1
        else:
            log_prob += log_prob_mat[pos[a]][pos[b]]
            count += 1

    return math.exp(log_prob / (count or 1))

def get_good_word_markov(accepted_words, pos, baseline_df):

    k = len(accepted_words)
    markov = [[10 for i in range(k)] for i in range(k)]

    for line in baseline_df['cleaned']:
        for a, b in ngram(2, line):
            markov[pos[a]][pos[b]] += 1

    for i, row in enumerate(markov):
        s = float(sum(row))
        for j in range(len(row)):
            if s == 0 or row[j] == 0:
                row[j] = row[j]
            else: 
                row[j] = math.log(row[j] / s)
            
    return markov

def get_bad_word_markov(accepted_words, pos, baseline_df, num_shuffles):
    k = len(accepted_words)
    markov = [[10 for i in range(k)] for i in range(k)]
    count = 0
    
    while count < num_shuffles:
        for line in baseline_df['cleaned']:
            hold = line.split()
            random.shuffle(hold)
            hold = " ".join(hold)
            for a, b in ngram(2, hold):
                markov[pos[a]][pos[b]] += 1
        
        for i, row in enumerate(markov):
            s = float(sum(row))
            for j in range(len(row)):
                if s == 0 or row[j] == 0:
                    row[j] = row[j]
                else: 
                    row[j] = math.log(row[j] / s) #log of element divided by sum of the row
        
        if count == 0:
            total_bad_markov = markov
        else:
            for i in range(len(markov)):
                for j in range(len(markov[i])):
                    total_bad_markov[i][j] += markov[i][j]
        
        markov = [[10 for i in range(k)] for i in range(k)]
        count += 1
    
    avg_markov = []
    hold = []
    for i in range(len(total_bad_markov)):
        for j in range(len(total_bad_markov[i])):
            hold.append(total_bad_markov[i][j] / num_shuffles)
        avg_markov.append(hold)
        hold = []
        
    return avg_markov

num_shuffles = 7
good_word_markov = get_good_word_markov(accepted_words, pos, baseline_df)
bad_word_markov = get_bad_word_markov(accepted_words, pos, baseline_df, num_shuffles)

def add_period_tag(corpus):
    txtblob_corpus = [TextBlob(corpus[i]) for i in range(len(corpus))]
    
    txtblob_sentences = []
    for i in range(len(txtblob_corpus)):
        txtblob_sentences.append(txtblob_corpus[i].sentences)
        
    sentence_tags = []
    temp = []
    for i in range(len(txtblob_sentences)):
        for j in range(len(txtblob_sentences[i])):
            temp.append(txtblob_sentences[i][j].tags)
        sentence_tags.append(temp)
        temp = []
        
    end = ('.', 'end')
    
    for i in range(len(sentence_tags)):
        for j in range(len(sentence_tags[i])):
            sentence_tags[i][j].append(end)
            
    return sentence_tags

baseline_corpus_tags = add_period_tag(baseline_corpus)
test_corpus_tags = add_period_tag(summaries)

def analyze_pos_data(fragments, fragment_tag, pos_names):

    enum_pos_tags = []
    hold = []
    for i in range(len(pos_names)):
        hold.append(pos_names[i])
        hold.append(i)
        enum_pos_tags.append(hold)
        hold = []
    
    pos_quantity = []
    pos_data = []
    order = []
    keep = len(enum_pos_tags)*[0]
    temp = []
    hold = []
    for i in range(len(fragment_tag)):
        for j in range(len(fragment_tag[i])):
            for k in range(len(enum_pos_tags)):
                if fragment_tag[i][j][1] == enum_pos_tags[k][0]:
                    keep[k] += 1
                    hold.append(fragment_tag[i][j][1])
                    temp.append(enum_pos_tags[k][1])
                else:
                    None
        pos_data.append(hold)
        pos_quantity.append(keep)
        keep = len(enum_pos_tags)*[0]
        order.append(temp)
        temp = []
        hold = []
        
    base_df = pd.DataFrame(pos_quantity, columns=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', 'end'])
    base_df.insert(0, "Fragments", fragments, True)
    
    pos_df = pd.DataFrame(pos_names, columns=['POS Names'])
    
    pos_array = [(pos, base_df[pos].sum()) for pos in pos_df['POS Names'].values]
    pos_names_ = list(map(lambda x: x[0], pos_array))
    pos_counts = list(map(lambda x: x[1], pos_array))
    
    graph_pos_df = pd.DataFrame({'name': pos_names_, 'count': pos_counts})
    dep_variable = np.arange(len(graph_pos_df['name'].values))
    plt.bar(dep_variable, graph_pos_df['count'].values)
    plt.xticks(dep_variable, graph_pos_df['name'].values, rotation='vertical')
    plt.title("Part of Speech Distribution")
    plt.xlabel("Parts of Speech")
    plt.ylabel("Number of Appearances")
        
    return pos_data, base_df

pos_names = ['CC','CD', 'DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', 'end']
   
for i in range(len(baseline_corpus_tags)):
    baseline_corpus_tags[i] = [j for sub in baseline_corpus_tags[i] for j in sub]

baseline_pos_data, base_df = analyze_pos_data(baseline_corpus, baseline_corpus_tags, pos_names)

def get_good_pos_markov(pos_data, pos_tags):
    ngrams = []
    for i in pos_data:
        ngrams.append(revised_ngram(2, i))
        
    flat_pos_ngram = [i for sub in ngrams for i in sub]
    pos = dict([(char, idx) for idx, char in enumerate(pos_tags)])

    markov = [[0 for i in range(len(pos_tags))] for i in range(len(pos_tags))]
    for i in range(len(flat_pos_ngram)):
        first_pos_element = pos[flat_pos_ngram[i][0]]
        second_pos_element = pos[flat_pos_ngram[i][1]]
        markov[first_pos_element][second_pos_element] += 1
        
    for i, row in enumerate(markov):
        s = float(sum(row))
        for j in range(len(row)):
            if s == 0 or row[j] == 0:
                row[j] = row[j]
            else:
                row[j] = math.log(row[j] / s)
    
    return markov

def get_bad_pos_markov(corpus, corpus_tags, pos_names, num_shuffles):
    shuff_corpus_tags = corpus_tags
    total_bad_markov = []
    count = 0
    while count < num_shuffles:
        for i in range(len(shuff_corpus_tags)):
            random.shuffle(shuff_corpus_tags[i])
        
        shuff_pos_data, data_frame = analyze_pos_data(corpus, shuff_corpus_tags, pos_names)
        bad_markov_model = get_good_pos_markov(shuff_pos_data, pos_names)
        
        if count == 0:
            total_bad_markov = bad_markov_model
        else:
            for i in range(len(bad_markov_model)):
                for j in range(len(bad_markov_model[i])):
                    total_bad_markov[i][j] += bad_markov_model[i][j]
        
        markov = [[0 for i in range(len(pos_names))] for i in range(len(pos_names))]
        count += 1
        
    avg_markov = []
    hold = []
    for i in range(len(total_bad_markov)):
        for j in range(len(total_bad_markov[i])):
            hold.append(total_bad_markov[i][j] / num_shuffles)
        avg_markov.append(hold)
        hold = []
        
    return avg_markov
               
good_pos_markov = get_good_pos_markov(baseline_pos_data, pos_names)
bad_pos_markov = get_bad_pos_markov(baseline_corpus, baseline_corpus_tags, pos_names, num_shuffles)

def visualize_word_matrix(markov, plot_name):  
    fig = plt.figure()
    
    ax1 = fig.add_subplot(122)
    ax1.imshow(markov, interpolation='nearest', cmap=plt.cm.Blues)
    
    plt.title("Log Probability of j-th Token Given i-th Token")
    plt.xlabel("Unique Token")
    plt.ylabel("Unique Token")
    
    heatmap = plt.pcolor(markov)
    plt.colorbar(heatmap, orientation='vertical', fraction=0.04)
    
    plt.savefig(plot_name, dpi = 400, bbox_inches='tight')
    
    plt.show()

def visualize_pos_matrix(markov, pos, base_df, plot_name):  
    fig = plt.figure()
    
    ax1 = fig.add_subplot(122)
    ax1.imshow(markov, interpolation='nearest', cmap=plt.cm.Blues)
    
    pos_df = pd.DataFrame(pos_names, columns=['POS Names'])
    
    pos_array = [(pos, base_df[pos].sum()) for pos in pos_df['POS Names'].values]
    pos_names_ = list(map(lambda x: x[0], pos_array))
    pos_counts = list(map(lambda x: x[1], pos_array))
    
    graph_pos_df = pd.DataFrame({'name': pos_names_, 'count': pos_counts})
    dep_variable = np.arange(len(graph_pos_df['name'].values))
    
    plt.xticks(dep_variable, graph_pos_df['name'].values, rotation='vertical')
    plt.yticks(dep_variable, graph_pos_df['name'].values)
    
    plt.tick_params(axis='x', which='major', labelsize=4.5)
    plt.tick_params(axis='y', which='major', labelsize=4.5)
    
    plt.title("Log Probability of j-th POS Given i-th POS")
    plt.xlabel("Parts of Speech")
    plt.ylabel("Parts of Speech")
    
    heatmap = plt.pcolor(markov)
    plt.colorbar(heatmap, orientation='vertical', fraction=0.04)
    
    plt.savefig(plot_name, dpi = 400, bbox_inches='tight')
    
    plt.show()
    
visualize_word_matrix(good_word_markov, 'Token Markov.png')
visualize_word_matrix(bad_word_markov, 'Average Token Markov.png')
visualize_pos_matrix(good_pos_markov, pos_names, base_df, 'POS Markov.png')
visualize_pos_matrix(bad_pos_markov, pos_names, base_df, 'Average POS Markov.png')

def compare_good_bad_pos_markov(summary_tag, good_markov, bad_markov, scale_factor, pos_names):
    pos = dict([(char, idx) for idx, char in enumerate(pos_names)])
    
    sensical_tags = []
    hold_good = []
    hold_bad = []
    good_prob = []
    bad_prob = []
    for i in range(len(summary_tag)):
        for j in range(len(summary_tag[i])):
            hold_tags = [TextBlob(summary_tag[i][j][0]).tags[k][1] for k in range(len(TextBlob(summary_tag[i][j][0]).tags))]
            hold_good.append(avg_pos_prob(hold_tags, good_markov, pos))
            hold_bad.append(avg_pos_prob(hold_tags, bad_markov, pos))
            if ((avg_pos_prob(hold_tags, good_markov, pos) > scale_factor*avg_pos_prob(hold_tags, bad_markov, pos)) == True):
                sensical_tags.append(summary_tag[i][j])
            else:
                None
            hold_tags = None
        good_prob.append(hold_good)
        bad_prob.append(hold_bad)
        hold_good = []
        hold_bad = []
                
    return sensical_tags, good_prob, bad_prob

def compare_good_bad_word_markov(summary_tag, good_markov, bad_markov, scale_factor, pos):
    pos = dict([(char, index) for index, char in enumerate(pos)])
    
    sensical_tags = []
    hold_good = []
    hold_bad = []
    good_prob = []
    bad_prob = []
    for i in range(len(summary_tag)):
        for j in range(len(summary_tag[i])):
            hold = summary_tag[i][j][0]
            hold_good.append(avg_word_prob(hold, good_markov, pos))
            hold_bad.append(avg_word_prob(hold, bad_markov, pos))
            if ((avg_word_prob(hold, good_markov, pos) > scale_factor*avg_word_prob(hold, bad_markov, pos)) == True):
                sensical_tags.append(summary_tag[i])
            else:
                None
            hold = None
        good_prob.append(hold_good)
        bad_prob.append(hold_bad)
        hold_good = []
        hold_bad = []
                
    return sensical_tags, good_prob, bad_prob

def compare_both_markov(summary_tag, good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names, word_sf, pos_sf):
    pos = dict([(char, index) for index, char in enumerate(pos)])
    pos_names = dict([(char, idx) for idx, char in enumerate(pos_names)])
    
    sensical_tags = []
    for i in range(len(summary_tag)):
        for j in range(len(summary_tag[i])):
            pos_sequence = [TextBlob(summary_tag[i][j][0]).tags[k][1] for k in range(len(TextBlob(summary_tag[i][j][0]).tags))]
            word_sequence = summary_tag[i][j][0]
            if((avg_word_prob(word_sequence, good_word_markov, pos)*avg_pos_prob(pos_sequence, good_pos_markov, pos_names)) > (pos_sf*word_sf*avg_word_prob(word_sequence, bad_word_markov, pos)*avg_pos_prob(pos_sequence, bad_pos_markov, pos_names))):
                sensical_tags.append(summary_tag[i][j])
            else:
                None
            pos_sequence = None
            word_sequence = None
    
    return sensical_tags

pos_sense, sen_prob_good_pos_markov, sen_prob_bad_pos_markov = compare_good_bad_pos_markov(summary_tag, good_pos_markov, bad_pos_markov, 1, pos_names)
word_sense, sen_prob_good_word_markov, sen_prob_bad_word_markov = compare_good_bad_word_markov(summary_tag, good_word_markov, bad_word_markov, 1, pos)

good_pos_avg = 0
good_word_avg = 0
bad_pos_avg = 0
bad_word_avg = 0
count = 0
for i in range(len(sen_prob_good_pos_markov)):
    for j in range(len(sen_prob_good_pos_markov[i])):
        good_pos_avg += sen_prob_good_pos_markov[i][j]
        good_word_avg += sen_prob_good_word_markov[i][j]
        bad_pos_avg += sen_prob_bad_pos_markov[i][j]
        bad_word_avg += sen_prob_bad_word_markov[i][j]
    count += len(sen_prob_good_pos_markov[i])

good_pos_avg = float(good_pos_avg/count)
good_word_avg = float(good_word_avg/count)
bad_pos_avg = float(bad_pos_avg/count)
bad_word_avg = float(bad_word_avg/count)

word_scale_factor = good_word_avg/bad_word_avg
pos_scale_factor =  good_pos_avg/bad_pos_avg

sensical_hmw = compare_both_markov(summary_tag, good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names, word_scale_factor, pos_scale_factor)

all_sensical_hmw = []
for i in range(len(sensical_hmw)):
    all_sensical_hmw.append(sensical_hmw[i][0])

final_hmw = []
hmw_with_summary = []
hold = []
count = 0
for i in all_sensical_hmw:
    if i not in hold:
        temp = 'How might we ' + i
        final_hmw.append(temp)
        hold.append(i)
        hmw_with_summary.append(sensical_hmw[count])
        count += 1
    else:
        count += 1

for j in range(len(final_hmw)-1, -1, -1):
    if TextBlob(final_hmw[j]).tags[-1][0] == 'i' or TextBlob(final_hmw[j]).tags[-1][1] == 'IN':
        del final_hmw[j]
    else:
        final_hmw[j] = final_hmw[j] + '?'

usergen_df = pd.read_csv('user_generated.csv')
user_generated = []
for i in usergen_df['user_generated']:
    user_generated.append(i)

full_txt = []
full_txt.append(user_generated)
full_txt.append(final_hmw)
full_txt = [i for sub in full_txt for i in sub]

hold = []
for i in full_txt:
    if i not in hold:
        hold.append(i)

full_txt = hold

with open("generated_hmw.txt", "w") as txt_file:
    for line in full_txt:
        txt_file.write("".join(line) + "\n")
        
def text_generator(path_to_file, EPOCHS): 

    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    print('Length of text: {} characters'.format(len(text)))
    
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))
    
    example_texts = ['abcdefg', 'xyz']
    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
    
    ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
    ids = ids_from_chars(chars)
    
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)
    chars = chars_from_ids(ids)
    
    tf.strings.reduce_join(chars, axis=-1).numpy()
    
    def text_from_ids(ids):
      return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
    
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    
    for ids in ids_dataset.take(10):
        print(chars_from_ids(ids).numpy().decode('utf-8'))
        
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    
    for seq in sequences.take(1):
      print(chars_from_ids(seq))
      
    for seq in sequences.take(5):
      print(text_from_ids(seq).numpy())
      
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    
    split_input_target(list("Tensorflow"))
    dataset = sequences.map(split_input_target)
    for input_example, target_example in  dataset.take(1):
        print("Input :", text_from_ids(input_example).numpy())
        print("Target:", text_from_ids(target_example).numpy())

    BATCH_SIZE = 8
    BUFFER_SIZE = 100
    
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024
    
    class MyModel(tf.keras.Model):
      def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True, 
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
      def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
          states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
    
        if return_state:
          return x, states
        else: 
          return x
    
    model = MyModel(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    mean_loss = example_batch_loss.numpy().mean()
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", mean_loss)
    tf.exp(mean_loss).numpy()
    model.compile(optimizer='adam', loss=loss)
    
    checkpoint_dir = './training_checkpoints'

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    
    class OneStep(tf.keras.Model):
      def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature=temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
    
        skip_ids = self.ids_from_chars(['','[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(

            values=[-float('inf')]*len(skip_ids),
            indices = skip_ids,

            dense_shape=[len(ids_from_chars.get_vocabulary())]) 
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
    
      @tf.function
      def generate_one_step(self, inputs, states=None):

        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
    
        predicted_logits, states =  self.model(inputs=input_ids, states=states, return_state=True)

        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature

        predicted_logits = predicted_logits + self.prediction_mask
    
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)
    
        return predicted_chars, states
    
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    
    states = None
    num_generations = 25
    temp = []
    for i in range(num_generations):
        temp.append('Generated Text: ')
    next_char = tf.constant(temp)
    result = [next_char]
    
    for n in range(5000):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)
    
    result = tf.strings.join(result)
    
    hold = []
    gen_text = []
    for i in range(len(result)):
      hold = str(result[i].numpy())
      hold = hold.splitlines()
      hold = str(hold).split('\\n')
      hold = hold[1:len(hold)-1]
      for j in range(len(hold)):
          hold[j] = hold[j][:-1]

      gen_text.append(hold)

    return gen_text

def compare_both_markov_no_tag(hmw, good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names, word_sf, pos_sf):
    pos = dict([(char, index) for index, char in enumerate(pos)])
    pos_names = dict([(char, idx) for idx, char in enumerate(pos_names)])
    
    sensical_tags = []
    for i in range(len(hmw)):
        pos_sequence = [TextBlob(hmw[i]).tags[j][1] for j in range(len(TextBlob(hmw[i]).tags))]
        word_sequence = hmw[i]
        good_prob = []
        bad_prob = []
        good_prob_val = avg_word_prob(word_sequence, good_word_markov, pos)*avg_pos_prob(pos_sequence, good_pos_markov, pos_names)
        bad_prob_val = avg_word_prob(word_sequence, bad_word_markov, pos)*avg_pos_prob(pos_sequence, bad_pos_markov, pos_names)
        if((good_prob_val) >= (pos_sf*word_sf*bad_prob_val)):
            sensical_tags.append(hmw[i])
            good_prob.append(good_prob_val)
            bad_prob.append(bad_prob_val)
        else:
            good_prob.append(good_prob_val)
            bad_prob.append(bad_prob_val)
        pos_sequence = None
        word_sequence = None
    
    return sensical_tags, good_prob, bad_prob

def recursive_generator(good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names, EPOCHS):
    path_to_file = 'generated_hmw.txt'
    gen_text = text_generator(path_to_file, EPOCHS)
    
    hmw = []
    for i in range(len(gen_text)):
        for j in range(len(gen_text[i])):
            hmw.append(" ".join(gen_text[i][j].split("?")[0].split()[3:]))
    
    gen_text, good_prob, bad_prob = compare_both_markov_no_tag(hmw, good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names, 2, 2)
    
    hmw_questions = []
    for i in range(len(gen_text)):
        temp = 'How might we ' + gen_text[i] + '?'
        hmw_questions.append(temp)
    
    full_txt = []
    full_txt.append(hmw_questions)
    full_txt.append(user_generated)
    full_txt = [i for sub in full_txt for i in sub]

    hold = []
    for i in full_txt:
      if i not in hold:
          hold.append(i)
    
    full_txt = hold
    
    with open("generated_hmw.txt", "a") as txt_file:
        for line in full_txt:
            txt_file.write("".join(line) + "\n")
            
    return gen_text, good_prob, bad_prob

cycles = 15
EPOCHS = 18
count = 0
len_tracker = []
keep_all = []
hmw_questions = []
store_good_prob = []
store_bad_prob = []
while count < cycles:
    EPOCHS = EPOCHS - count
    hmw_questions, good_prob, bad_prob = recursive_generator(good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names, EPOCHS)
    keep_all.append(hmw_questions)
    len_tracker.append(len(hmw_questions))
    store_good_prob.append(good_prob)
    store_bad_prob.append(bad_prob)
    count += 1

import pandas as pd
import os

for i in range(len(keep_all)):
    write_to_csv = []
    for j in range(len(keep_all[i])):
        keep_all[i][j] = "How might we " + keep_all[i][j] + "?"
        write_to_csv.append(keep_all[i][j])
    write_df = pd.DataFrame(write_to_csv, columns=['Generated HMW Question'])
    file_name = 'Neural Network Iteration ' + str(i+1) +".csv"
    write_df.to_csv(file_name)
    
def avg_probs(hmw, good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names):
    pos = dict([(char, index) for index, char in enumerate(pos)])
    pos_names = dict([(char, idx) for idx, char in enumerate(pos_names)])
    
    avg_prob_list = []
    hold = []
    good_prob = []
    bad_prob = []
    store_good_prob = []
    store_bad_prob = []
    avg_good_prob = 0
    avg_bad_prob = 0
    for i in range(len(hmw)):
        for j in range(len(hmw[i])):
            pos_sequence = [TextBlob(hmw[i][j]).tags[k][1] for k in range(len(TextBlob(hmw[i][j]).tags))]
            word_sequence = hmw[i][j]
            g_temp = avg_word_prob(word_sequence, good_word_markov, pos)*avg_pos_prob(pos_sequence, good_pos_markov, pos_names)
            avg_good_prob += g_temp
            good_prob.append(g_temp)
            b_temp = avg_word_prob(word_sequence, bad_word_markov, pos)*avg_pos_prob(pos_sequence, bad_pos_markov, pos_names)
            avg_bad_prob += b_temp
            bad_prob.append(b_temp)
        avg_good_prob = avg_good_prob/len(hmw[i])
        avg_bad_prob = avg_bad_prob/len(hmw[i])
        hold.append(avg_good_prob)
        hold.append(avg_bad_prob)
        store_good_prob.append(good_prob)
        store_bad_prob.append(bad_prob)
        good_prob = []
        bad_prob = []
        avg_prob_list.append(hold)
        hold = []
        avg_good_prob = 0
        avg_bad_prob = 0
    
    return avg_prob_list, store_good_prob, store_bad_prob

prob_avgs = []
prob_avgs, good_prob, bad_prob = avg_probs(keep_all, good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names)
print(prob_avgs)

def compare_both_markov_no_tag(hmw, good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names, s):
    pos = dict([(char, index) for index, char in enumerate(pos)])
    pos_names = dict([(char, idx) for idx, char in enumerate(pos_names)])
    
    sensical_tags = []
    for i in range(len(hmw)):
        pos_sequence = [TextBlob(hmw[i]).tags[j][1] for j in range(len(TextBlob(hmw[i]).tags))]
        word_sequence = hmw[i]
        if((avg_word_prob(word_sequence, good_word_markov, pos)*avg_pos_prob(pos_sequence, good_pos_markov, pos_names)) >= (s*avg_word_prob(word_sequence, bad_word_markov, pos)*avg_pos_prob(pos_sequence, bad_pos_markov, pos_names))):
            sensical_tags.append(hmw[i])
        else:
            None
        pos_sequence = None
        word_sequence = None
    
    return sensical_tags

finalized_hmw = compare_both_markov_no_tag(keep_all[4], good_pos_markov, bad_pos_markov, pos, good_word_markov, bad_word_markov, pos_names, 20)
print(len(finalized_hmw))
print(finalized_hmw)

def plot_prob(test, good_prob, bad_prob, y_max):
    x = range(0,len(test))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    count = 0
    x = []
    y_shuff = []
    y_unshuff = []
    yerror = []
    yerrors = []
    
    questions_df = pd.DataFrame()
    
    for i in range(len(test)):
        y_unshuff.append(test[i][0])
        y_shuff.append(test[i][1])
        count += 1
        x.append(str(count))
        mean_y = np.mean(good_prob[i])
        stdev_y = np.std(good_prob[i])
        sterr_y = stdev_y/math.sqrt(len(good_prob[i]))
        yerror.append(2*sterr_y)
        mean_ys = np.mean(bad_prob[i])
        stdev_ys = np.std(bad_prob[i])
        sterr_ys = stdev_ys/math.sqrt(len(bad_prob[i]))
        yerrors.append(2*sterr_ys)

    xAxis = [i + 0.5 for i, _ in enumerate(x)]
    ax1.bar(xAxis, y_unshuff, color='teal', label = 'Avg Prob Evaluated on Unshuffled')
    ax1.bar(xAxis, y_shuff, color='red', width = 0.6, label = 'Avg Prob Evaluated on Shuffled')
    ax1.set_ylim([0,y_max])
    plt.errorbar(xAxis, y_unshuff, yerr=yerror, fmt='k^',capsize=5,capthick=2,ms=3,markerfacecolor='none',mew=2)
    plt.xticks([i + 0.5 for i, _ in enumerate(x)], x)
    plt.xlabel('Cycle')
    plt.ylabel('Average Probability')
    plt.legend(loc='upper left');
    plot_name = 'Shuffled_Unshuffled Probabilities'
    plt.savefig(plot_name, dpi = 400, bbox_inches='tight')
    plt.show()

    return yerror

y_max = 0.002
yerror = plot_prob(prob_avgs, good_prob, bad_prob, y_max)
print(yerror)
