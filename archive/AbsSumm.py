# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:58:14 2020

@author: Emmett
"""

## UNSUPERVISED ABSTRACTIVE - TRANSFER LEARNING APPLICATION #

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')
"""
text =
The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
"We'll be the comeback kids, all of us," he said. "We want to get our country back."
The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.

#preprocess_text = text.strip().replace("\n","")
#t5_prepared_Text = "summarize: "+ preprocess_text
#print ("original text preprocessed: \n", preprocess_text)
"""

import numpy as np
import pandas as pd
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
stoplist = stopwords.words('english')

def text_cleaner(text,num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stoplist]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1: #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

data = pd.read_csv("keyword_comment_cleaned.csv", encoding = 'utf-8')

#removes duplicated rows of 'text' column
data.drop_duplicates(subset=['comment_body'], inplace=True)
#drops rows that contain missing values
data.dropna(axis=0, inplace=True)

clean_sentences = []
for i in data['comment_body']:
    clean_sentences.append(text_cleaner(i, 0))
    
data['clean_sentences'] = clean_sentences

t5_prepared_Text = []
for i in range(len(clean_sentences)):
    temp = "summarize: " + clean_sentences[i]
    t5_prepared_Text.append(temp)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

#summmarize 
summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#print ("\n\nSummarized text: \n",output)