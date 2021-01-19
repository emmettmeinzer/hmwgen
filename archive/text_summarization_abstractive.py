# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:41:10 2020

@author: Emmett
"""
#ABSTRACTIVE SUMMARIZATION
import pandas as pd
import numpy as py

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)

#Wikipedia exerpt test
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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
post = []
for i in data['comment_body']:
    sentences.append(i)
for i in data['post_id']:
    post.append(i)
    
i = 0
_id = ''
conversations = []
hold = []
while i < len(post):
    _id = post[i]
    while _id == post[i]:
        hold.append(sentences[i])
        i += 1
        if i == len(post):
            break
    conversations.append(hold)
    hold = []

text = [' '.join(conversations[i]) for i in range(len(conversations))]

#max_length/min_length: min/max number of tokens in summary
#length_penalty: penalizes model for producing summary above/below the maximum and minimum length thresholds
#num_beams: numbers of beams that explore potential token for most promising predictions
loop_len = len(text)
summaries = []
for i in range(loop_len):
    inputs = tokenizer.encode("summarize: " + text[i], return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=25, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
    summary = summary.split(' ', 1)[1]
    summaries.append(summary)
    inputs = None
    summary_ids = None
    summary = None

file = open("t5_summaries.txt","w")
for i in range(len(summaries)):
    file.write(summaries[i])
    file.write("\n")
        