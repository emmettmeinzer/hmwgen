# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:34:06 2021

@author: Emmett
"""
import warnings
warnings.filterwarnings("ignore")

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

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
stoplist = stopwords.words('english')

def text_cleaner(text,num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    #newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
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

data = pd.read_csv('Reviews.csv')

#removes duplicated rows of 'text' column
data.drop_duplicates(subset=['Text'], inplace=True)
#drops rows that contain missing values
data.dropna(axis=0, inplace=True)

clean_sentences = []
for i in data['Text']:
    clean_sentences.append(text_cleaner(i, 0))

clean_summary = []
for i in data['Summary']:
    clean_summary.append(text_cleaner(i, 1))

data['clean_sentences'] = clean_sentences
data['clean_summary'] = clean_summary

#replaces empty entries with NaN value
data.replace('', np.nan, inplace=True)
#drops rows that contain missing values
data.dropna(axis=0, inplace=True)

text_word_count = []
summary_word_count = []

#populate list with length of every cleaned text sample
for i in data['clean_sentences']:
      text_word_count.append(len(i.split()))
#populate list with length of every cleaned summary
for i in data['clean_summary']:
      summary_word_count.append(len(i.split()))
#create a data structure with 'text' column of text sample lengths and 'summary' column of summary lengths
length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

#create historgrams of summary and text lengths (parse data into 30 bins)
length_df.hist(bins = 30)
plt.show()
"""
#find number of words in 95% of the cleaned summaries 
for length_summary in range(100):
    count = 0
    for i in data['clean_summary']:
        if(len(i.split())<=length_summary):
            count += 1
    ratio_lessthan = count/len(data['clean_summary'])
    if ratio_lessthan >= 0.95:
        print("95% of summaries have less than " + str(length_summary) + " words.")
        break
#Output: 95% of summaries have less than 9 words.

#find number of words in 95% of the cleaned text samples
for length_text in range(100,501):
    count = 0
    for i in data['clean_sentences']:
        if(len(i.split())<=length_text):
            count += 1
    ratio_lessthan = count/len(data['clean_sentences'])
    if ratio_lessthan >= 0.95:
        print("95% of text samples have less than " + str(length_text) + " words.")
        break
#Output: 95% of text samples have less than 104 words.
"""
#95% length search defines the values of max text length and max summary length
max_text_len=103
max_summary_len=8
#max_text_len = (length_text-1)
#max_summary_len = (length_summary-1)

#convert lists into arrays
clean_sentences = np.array(data['clean_sentences'])
clean_summary = np.array(data['clean_summary'])

short_text = []
short_summary = []

#store all text samples and summaries with lengths less than defined maximums
for i in range(len(clean_sentences)):
    if(len(clean_summary[i].split())<=max_summary_len and len(clean_sentences[i].split())<=max_text_len):
        short_text.append(clean_sentences[i])
        short_summary.append(clean_summary[i])

#create a data structure with 'text' column of 95% text sample lengths and 'summary' column of 95% summary lengths
df=pd.DataFrame({'text':short_text,'summary':short_summary})

#add sostok and eostok to beginning and end of each summary
df['summary'] = df['summary'].apply(lambda x : 'sostok ' + x + ' eostok')

"""
REDDIT VALIDATION DATA PREPROCESSING AND MANIPULATION
reddit_val_data = pd.read_csv("keyword_comment_cleaned.csv", encoding = 'utf-8')
reddit_val_data.drop_duplicates(subset=['comment_body'], inplace=True)
reddit_val_data.dropna(axis=0, inplace=True)
clean_val_sentences = []
for i in reddit_val_data['comment_body']:
    clean_val_sentences.append(sentence_cleaner(i))
clean_val_summary = []
for i in reddit_val_data['val_summary']:
    clean_val_summary.append(sentence_cleaner(i))
reddit_val_data['clean_val_sentences'] = clean_val_sentences
reddit_val_data['clean_val_summary'] = clean_val_summary
reddit_val_data.replace('', np.nan, inplace=True)
reddit_val_data.dropna(axis=0, inplace=True)
valtext_word_count = []
valsummary_word_count =[]
for i in reddit_val_data['clean_val_sentences']:
    valtext_word_count.append(len(i.split()))
for i in reddit_val_data['clean_val_summary']:
    valsummary_word_count.append(len(i.split()))
length_valdf = pd.DataFrame({'comment_body':valtext_word_count, 'val_summary':valsummary_word_count})
cleaned_val_sentences = np.array(reddit_val_data['clean_val_sentences'])
cleaned_val_summary = np.array(reddit_val_data['clean_val_summary'])
short_val_sentences = []
short_val_summary = []
for i in range(len(clean_val_sentences)):
    if(len(clean_val_summary[i].split()) <= max_summary_len and len(clean_val_sentences[i].split())<=max_text_len):
        short_val_sentences.append(clean_val_sentences[i])
        short_val_summary.append(clean_val_summary[i])
val_df = pd.DataFrame({'comment_body':short_val_sentences, 'val_summary':short_val_summary})
val_df['val_summary'] = val_df['val_summary'].apply(lambda x : 'sostok ' + x + ' eostok')
x_tr = df['text']
y_tr = df['summary']
x_val = val_df['comment_body']
y_val = val_df['val_summary']
"""

#split dataset into 90% length training data and 10% validation data
from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']),np.array(df['summary']),test_size=0.1,random_state=0,shuffle=True)
#x_tr: text samples of training data
#x_val: text samples of validation data
#y_tr: summaries of training data
#y_val: summaries of validation data

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

#rare words defined as words that appear less than 4 times through the samples
thresh=4 #arbitrarily set as far as I can tell
#tot_count: size of vocabulary (every unique words in the text)
#count: number of rare words whose count falls below threshold
#(tot_count - count): top most common words
count=0
tot_count=0
freq=0
tot_freq=0
for key,value in x_tokenizer.word_counts.items():
    tot_count += 1
    tot_freq=tot_freq+value
    if(value<thresh):
        count += 1
        freq=freq+value
    
print("% of rare words in vocabulary:",(count/tot_count)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)

#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=tot_count - count) 
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences
x_tr_seq = x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

#pad arrays with zeros up to maximum length value
x_tr = pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

#size of vocabulary (+1 for padding token)
x_voc   =  x_tokenizer.num_words + 1
print(x_voc)

##REPEAT PROCESS FOR SUMMARIES##
#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_tr))

thresh=6 #arbitrarily set as far as I can tell
count=0
tot_count=0
freq=0
tot_freq=0

for key,value in y_tokenizer.word_counts.items():
    tot_count += 1
    tot_freq=tot_freq+value
    if(value<thresh):
        count += 1
        freq=freq+value
    
print("% of rare words in vocabulary:",(count/tot_count)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)

#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=tot_count-count) 
y_tokenizer.fit_on_texts(list(y_tr))

#convert text sequences into integer sequences
y_tr_seq = y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq = y_tokenizer.texts_to_sequences(y_val) 

#pad arrays with zeros up to maximum length value
y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

#size of vocabulary (+1 for padded token)
y_voc  =   y_tokenizer.num_words + 1
print(y_voc)
#check for same lengths
#if same, we know sostok is sufficient unique as a start token such that it doesn't appear in any of the samples
print(y_tokenizer.word_counts['sostok'], len(y_tr))

#if there are array elements with just the start and end tokens, then remove them
#removes any empty summaries
ind=[]
for i in range(len(y_tr)):
    count=0
    for j in y_tr[i]:
        if j != 0:
            count += 1
    if(count == 2):
        ind.append(i)
y_tr=np.delete(y_tr,ind, axis=0)
#remove text samples corresponding to empty summaries
x_tr=np.delete(x_tr,ind, axis=0)

#do the same for the validation dataset
ind=[]
for i in range(len(y_val)):
    count = 0
    for j in y_val[i]:
        if j != 0:
            count += 1
    if(count == 2):
        ind.append(i)
y_val=np.delete(y_val,ind, axis=0)
#remove text samples corresponding to empty summaries
x_val=np.delete(x_val,ind, axis=0)

from keras import backend as K
#resets all states generated by Keras
K.clear_session()

#latent vector has 300 dimensions (maybe arbitrary??)
latent_dim = 300
#embedding vector has 100 dimensions (maybe arbitrary??)
embedding_dim = 100

#Encoder
encoder_inputs = Input(shape=(max_text_len,))

#Embedding Layer
enc_emb = Embedding(x_voc, embedding_dim, trainable = True)(encoder_inputs)

#encoder lstm 1 (first layer in stack)
#return_sequence = True: when the return sequences parameter is set to True, LSTM produces the hidden state and cell state for every timestep
#dropout: tuneable parameter for Keras tuner (currently 0.4; I've seen 0.1 and 0.2 before, as well)
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#encoder lstm 2 (second layer in stack)
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm 3 (third layer in stack)
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#dense layer
#softmax activation (also seen 'tanh' and 'relu')
#tanh: tanh(x) - looks like an 's' curve
#relu: x = 0 for x <= 0 and x = x for x > 0
#softmax: outputs vector that represents the probability distributions of a list of potential outcomes (always sums to one)
decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#build up dictionary to convert the index to word for target and source vocabulary??
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

for i in range(0,100):
    print("Review:",seq2text(x_tr[i]))
    print("Original summary:",seq2summary(y_tr[i]))
    print("Predicted summary:",decode_sequence(x_tr[i].reshape(1,max_text_len)))
    print("\n")