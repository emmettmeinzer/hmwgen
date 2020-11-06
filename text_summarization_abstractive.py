# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:41:10 2020

@author: Emmett
"""
#ABSTRACTIVE SUMMARIZATION
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from attention import AttentionLayer

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

def sentence_cleaner(text):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [word for word in newString.split() if not word in stoplist]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

def summary_cleaner(text):
    newString = re.sub('"','', text)
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = newString.lower()
    tokens=newString.split()
    newString=''
    for i in tokens:
        if len(i)>1:                                 
            newString=newString+i+' '  
    return newString

data = pd.read_csv('Reviews.csv', nrows = 10000)
test_data = pd.read_csv('keyword_comment_cleaned.csv')

data.drop_duplicates(subset=['Text'], inplace=True)
data.dropna(axis=0, inplace=True)

clean_sentences = []
for i in data['Text']:
    clean_sentences.append(sentence_cleaner(i))  
clean_summary = []
for i in data['Summary']:
    clean_summary.append(summary_cleaner(i))

data['clean_sentences'] = clean_sentences
data['clean_summary'] = clean_summary
data['clean_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)

data['clean_summary'] = data['clean_summary'].apply(lambda x : '<START>' + x + '<END>')

test_data.drop_duplicates(subset=['comment_body'], inplace=True)
test_data.dropna(axis=0, inplace=True)

test_sentences = []
for i in test_data['comment_body']:
    test_sentences.append(sentence_cleaner(i))  
test_summary = []
for i in test_data['test_summary']:
    test_summary.append(summary_cleaner(i))

test_data['test_sentences'] = test_sentences
test_data['test_summary'] = test_summary
test_data['test_summary'].replace('', np.nan, inplace=True)
test_data.dropna(axis=0, inplace=True)

test_data['test_summary'] = test_data['test_summary'].apply(lambda x : '<START>' + x + '<END>')
    
#x_tr, x_val, y_tr, y_val = train_test_split(data['clean_sentences'], data['clean_summary'], test_size = 0, random_state = 0, shuffle = True)
x_tr = data['clean_sentences']
y_tr = data['clean_summary']
x_val = test_data['test_sentences']
y_val = test_data['test_summary']

max_len_text=80 
max_len_summary=10
    
#TEXT TOKENIZER
#prepare tokenizer for reviews
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))
#convert text sequencess into integer sequences
x_tr = x_tokenizer.texts_to_sequences(x_tr)
x_val = x_tokenizer.texts_to_sequences(x_val)
#padding zero up to maximum length
x_tr = pad_sequences(x_tr, maxlen = max_len_text, padding = 'post')
x_val = pad_sequences(x_val, maxlen = max_len_text, padding = 'post')

x_voc_size = len(x_tokenizer.word_index) + 1

#prepare tokenizer for summary
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))
#convert summary sequences into integer sequences
y_tr    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val   =   y_tokenizer.texts_to_sequences(y_val) 
#padding zero up to maximum length
y_tr    =   pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
y_val   =   pad_sequences(y_val, maxlen=max_len_summary, padding='post')

y_voc_size  =   len(y_tokenizer.word_index) + 1

#Long Short Term Memory
from keras import backend as k
k.clear_session()
latent_dim = 500

#encoder initialize
encoder_inputs = Input(shape=(max_len_text,))
enc_emb = Embedding(x_voc_size, latent_dim, trainable = True)(encoder_inputs)

#LSTM 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state = True)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#LSTM 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state = True)
encoder_output2, state_h2, state_c2 = encoder_lstm1(encoder_output1)

#LSTM 3
encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state = True)
encoder_outputs, state_h, state_c = encoder_lstm1(encoder_output2)

#decoder set up
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable = True)
dec_emb = dec_emb_layer(decoder_inputs)

#LSTM using encoder states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state = [state_h, state_c])

#Attention layer
attn_layer = AttentionLayer(name= 'attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

#concatenate attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation = 'softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

#Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:], epochs=1,callbacks=[es],batch_size=512, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

"""
from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend()
pyplot.show()
"""

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

#encoder inference
encoder_model = Model(inputs=encoder_inputs, outputs = [encoder_outputs, state_h, state_c])

#decoder inference
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_text, latent_dim))

#embedding of decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)

#set initial states to states from previous time step to predict next word
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

#dense softmax layer to generat probability distribution over the target vocabulary
decoder_output2 = decoder_dense(decoder_inf_concat)

#final decoder model
decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])

def decode_sequence(input_seq):
    #encode input as state vectors
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    #generate empty target sequence of length one
    target_seq = np.zeros((1,1))
    
    #chose start word as first word of target sequence
    target_seq[0,0] = target_word_index['start']
    
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        #sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token != 'end'):
            decoded_sentence += ' ' + sampled_token
            #exit condition
            if(sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary - 1)):
                stop_condition = True
        
        target_seq = np.zeros((1,1))
        target_seq[0,0] = sampled_token_index
        
        #update internal states
        e_h = h
        e_c = c
    return decoded_sentence

def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if((i != 0 and i != target_word_index['start']) and i != target_word_index['end']):
            newString = newString + reverse_target_word_index[i] + ' '
    return newString

def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if(i != 0):
            newString = newString + reverse_source_word_index[i] + ' '
    return newString

for i in range(len(x_val)):
    print("Review: ", seq2text(x_val[i]))
    print("Original summary: ", decode_sequence(x_val[i].reshape(1, max_len_text)))
    print("\n")