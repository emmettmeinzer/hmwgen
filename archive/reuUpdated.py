# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:41:14 2019

@author: Emmett & Binyang
"""

from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

##Let’s first build a corpus to train our tokenizer on. We’ll use stuff available in NLTK:

from nltk.corpus import gutenberg

# print (dir(gutenberg))
# print (gutenberg.fileids())

text = ""
for file_id in gutenberg.fileids():
    text += gutenberg.raw(file_id)
 
print (len(text))

##a funtion that converts a list to a string
def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1

##extract sentences from samples for following sentiment analysis
sampNum = 1
sent_df = pd.DataFrame()
i = 0

while (sampNum < 186):
    fileOpen = open("sample"+str(sampNum)+".txt","r")
    temp = fileOpen.readlines()
    temp = listToString(temp)
    
    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train(text)
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    
    ##Adding more abbreviations
    tokenizer._params.abbrev_types.add('dr')
 
    sent = tokenizer.tokenize(temp)
    
    for sent in sent:
        sent_df.loc[i, 'sent'] = sent
        sent_df.loc[i, 'sample'] = sampNum
        i += 1
    
    sampNum += 1

##NLTK’s built-in Vader Sentiment Analyzer will simply rank a piece of text as positive, negative or neutral 
##using a lexicon of positive and negative words.

##We can utilize this tool by first creating a Sentiment Intensity Analyzer (SIA) to categorize our headlines, 
##then we'll use the polarity_scores method to get the sentiment.

##We'll append each sentiment dictionary to a results list, which we'll transform into a dataframe:

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for idx, row in sent_df.iterrows():
    line = row['sent']
    score = sia.polarity_scores(line)
    sent_df.loc[idx, 'neg'] = score.get('neg')
    sent_df.loc[idx, 'neu'] = score.get('neu')
    sent_df.loc[idx, 'pos'] = score.get('pos')
    sent_df.loc[idx, 'compound'] = score.get('compound')

# pprint(results[:10], width=100)

##We will consider posts with a compound value greater than 0.2 as positive and less than -0.2 as negative. 
##There's some testing and experimentation that goes with choosing these ranges, and there is a trade-off to be 
##made here. If you choose a higher value, you might get more compact results (less false positives and false 
##negatives), but the size of the results will decrease significantly.

sent_df['label'] = 0
sent_df.loc[sent_df['compound'] > 0.3, 'label'] = 1
sent_df.loc[sent_df['compound'] < -0.3, 'label'] = -1
# sent_df.head()

##We have all the data we need to save, so let's do that:

sent_df.to_csv('sentiment analysis.csv', mode='a', encoding='utf-8', index=False)

##We can now keep appending to this csv, but just make sure that if you reassign the headlines set, you could get 
##duplicates. Maybe add a more advanced saving function that reads and removes duplicates before saving.

#Let's first take a peak at a few positive and negative headlines:

print("Positive headlines:\n")
pprint(list(sent_df[sent_df['label'] == 1].sent)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(sent_df[sent_df['label'] == -1].sent)[:5], width=200)

##Now let's check how many total positives and negatives we have in this dataset:

print(sent_df.label.value_counts())
print(sent_df.label.value_counts(normalize=True) * 100)

##The first line gives us raw value counts of the labels, whereas the second line provides percentages 
##with the normalize keyword.

##For fun, let's plot a bar chart:
"""
fig, ax = plt.subplots(figsize=(8, 8))

counts = sent_df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()
"""

##filter the sentences by number of words in it
for idx, row in sent_df.iterrows():
    sentence = row['sent']
    sent_df.loc[idx, 'len_sent'] = len(sentence.split())

##split positive and other sentences
pos = sent_df[sent_df['label'] == 1]
neg = sent_df[sent_df['label'] != 1]

import gensim
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text

corpus_full = []
for idx, row in sent_df.iterrows():
    temp = row['sent']
    temp1 = strip_non_alphanum(str(temp))
    temp2 = strip_punctuation(temp1)
    temp3 = strip_multiple_whitespaces(temp2)
    final = stem_text(temp3)
    corpus_full.append(final)

corpus_pos = []
for idx, row in pos.iterrows():
    temp = row['sent']
    temp1 = strip_non_alphanum(str(temp))
    temp2 = strip_punctuation(temp1)
    temp3 = strip_multiple_whitespaces(temp2)
    final = stem_text(temp3)
    corpus_pos.append(final)
    
corpus_neg = []
for idx, row in neg.iterrows():
    temp = row['sent']
    temp1 = strip_non_alphanum(str(temp))
    temp2 = strip_punctuation(temp1)
    temp3 = strip_multiple_whitespaces(temp2)
    final = stem_text(temp3)
    corpus_neg.append(final)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

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

print (len(stoplist))
stoplist.update(stop_words)

print(len(stop_words))
print(len(stoplist))

#standardize text -- makes all characters lowercase and removes common stop words
text_full = [[word for word in document.lower().split() if word not in stoplist]
        for document in corpus_full]
print(text_full)
text_pos = [[word for word in document.lower().split() if word not in stoplist]
        for document in corpus_pos]
text_neg = [[word for word in document.lower().split() if word not in stoplist]
        for document in corpus_neg]

#count number of times that word appears in corpus
#pair frequency with respective word in new array
from collections import defaultdict

frequency = defaultdict(int)
for text in text_full:
    for token in text:
        frequency[token] += 1

corpus_removeOne_full = [[token for token in text if frequency[token]>1] for text in text_full]

frequency = defaultdict(int)
for text in text_pos:
    for token in text:
        frequency[token] += 1
        
corpus_removeOne_pos = [[token for token in text if frequency[token]>1] for text in text_pos]

frequency = defaultdict(int)
for text in text_neg:
    for token in text:
        frequency[token] += 1
        
corpus_removeOne_neg = [[token for token in text if frequency[token]>1] for text in text_neg]


from gensim import corpora
#add corpora to dictionary
dictionary_full = corpora.Dictionary(corpus_removeOne_full)
dictionary_pos = corpora.Dictionary(corpus_removeOne_pos)
dictionary_neg = corpora.Dictionary(corpus_removeOne_neg)
#save dictionary for future reference
dictionary_full.save('redditTest_full.dict')
dictionary_pos.save('redditTest_pos.dict') #location of document in computer
dictionary_neg.save('redditTest_neg.dict')
#dict = gensim.corpora.Dictionary.load('redditTest.dict')

#assign numeric id to each token in dictionary
dictID_full = dictionary_full.token2id
dictID_pos = dictionary_pos.token2id
dictID_neg = dictionary_neg.token2id

#remove empty sentences
for text in corpus_removeOne_full:
    if len(text) == 0:
        corpus_removeOne_full.remove(text)

for text in corpus_removeOne_pos:
    if len(text) == 0:
        corpus_removeOne_pos.remove(text)
        
for text in corpus_removeOne_neg:
    if len(text) == 0:
        corpus_removeOne_neg.remove(text)


#converts each word into vector following same process as example
#Bag of Word Corpus of Full Sentiment
bow_corpus_full = [dictionary_full.doc2bow(text) for text in corpus_removeOne_full]
corpora.MmCorpus.serialize('redditTest_full.mm', bow_corpus_full)
corp_full = gensim.corpora.MmCorpus('redditTest_full.mm')

from gensim import models
tfidf_pos = models.TfidfModel(bow_corpus_full)
corpus_tfidf_full = tfidf_pos[bow_corpus_full]

#Bag of Word Corpus of Positive Sentiment
bow_corpus_pos = [dictionary_pos.doc2bow(text) for text in corpus_removeOne_pos]
corpora.MmCorpus.serialize('redditTest_pos.mm', bow_corpus_pos)
corp_pos = gensim.corpora.MmCorpus('redditTest_pos.mm')

from gensim import models
tfidf_pos = models.TfidfModel(bow_corpus_pos)
corpus_tfidf_pos = tfidf_pos[bow_corpus_pos]

#Bag of Word Corpus of Negative Sentiment
bow_corpus_neg = [dictionary_neg.doc2bow(text) for text in corpus_removeOne_neg]
corpora.MmCorpus.serialize('redditTest_neg.mm', bow_corpus_neg)
corp_neg = gensim.corpora.MmCorpus('redditTest_neg.mm')

from gensim import models
tfidf_neg = models.TfidfModel(bow_corpus_neg)
corpus_tfidf_neg = tfidf_neg[bow_corpus_neg]


#LDA Mallet for full corpus
mallet_path = '/Users/emmet/.spyder-py3-dev/REU_Project/mallet-2.0.8/bin/mallet'
lda_full = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus_full, num_topics=9, id2word=dictionary_full, workers=1, alpha=110, random_seed=109, iterations=50)
corpus_LDA_full = lda_full[bow_corpus_full]
lda_full.print_topics(9)

#LDA Mallet for positive corpus
mallet_path = '/Users/emmet/.spyder-py3-dev/REU_Project/mallet-2.0.8/bin/mallet'
lda_pos = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus_pos, num_topics=9, id2word=dictionary_pos, workers=1, alpha=110, random_seed=109, iterations=50)
corpus_LDA_pos = lda_pos[bow_corpus_pos]
lda_pos.print_topics(9)

#LDA Mallet for negative corpus
mallet_path = '/Users/emmet/.spyder-py3-dev/REU_Project/mallet-2.0.8/bin/mallet'
lda_neg = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus_neg, num_topics=9, id2word=dictionary_neg, workers=1, alpha=110, random_seed=109, iterations=50)
corpus_LDA_neg = lda_neg[bow_corpus_neg]
lda_neg.print_topics(9)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

colors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

#t-SNE plot for full corpus
n_topics = 9
topic_weights_full = []
for row_list in lda_full[bow_corpus_full]:
    tmp = np.zeros(n_topics)
    for i, w in row_list:
        tmp[i] = w
    topic_weights_full.append(tmp)
    
arr_full = pd.DataFrame(topic_weights_full).fillna(9).values
topic_num_full = np.argmax(arr_full, axis=1)
tsne_model_full = TSNE(n_components=3, random_state=None, method='barnes_hut', 
                  angle=0.5, init='pca')
tsne_lda_full = tsne_model_full.fit_transform(arr_full)

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.xlabel('t-SNE1'.translate(sub))
plt.ylabel('t-SNE2'.translate(sub))
plt.title('t-SNE Plot of Topics within Positive Sentiment Corpus')
tsne_full = plt.scatter(x=tsne_lda_full[:,0], y=tsne_lda_full[:,1])
plt.show(tsne_full)

"""
#t-SNE plot for positive corpus
n_topics = 9
topic_weights_pos = []
for row_list in lda_pos[bow_corpus_pos]:
    tmp = np.zeros(n_topics)
    for i, w in row_list:
        tmp[i] = w
    topic_weights_pos.append(tmp)
    
arr_pos = pd.DataFrame(topic_weights_pos).fillna(0).values
topic_num_pos = np.argmax(arr_pos, axis=1)
tsne_model_pos = TSNE(n_components=3, random_state=None, method='barnes_hut', 
                  angle=0.5, init='pca')
tsne_lda_pos = tsne_model_pos.fit_transform(arr_pos)

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.xlabel('t-SNE1'.translate(sub))
plt.ylabel('t-SNE2'.translate(sub))
plt.title('t-SNE Plot of Topics within Positive Sentiment Corpus')
tsne_pos = plt.scatter(x=tsne_lda_pos[:,0], y=tsne_lda_pos[:,1])
#plt.show(tsne_pos)


#t-SNE plot for negative corpus
n_topics = 9
topic_weights_neg = []
for row_list in lda_neg[bow_corpus_neg]:
    tmp = np.zeros(n_topics)
    for i, w in row_list:
        tmp[i] = w
    topic_weights_neg.append(tmp)
    
arr_neg = pd.DataFrame(topic_weights_neg).fillna(0).values
topic_num_neg = np.argmax(arr_neg, axis=1)
tsne_model_neg = TSNE(n_components=3, random_state=None, method='barnes_hut', 
                  angle=0.5, init='pca')
tsne_lda_neg = tsne_model_neg.fit_transform(arr_neg)

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.xlabel('t-SNE1'.translate(sub))
plt.ylabel('t-SNE2'.translate(sub))
plt.title('t-SNE Plot of Topics within Negative Sentiment Corpus')
tsne_neg = plt.scatter(tsne_lda_neg[:,0], tsne_lda_neg[:,1])
#plt.show(tsne_neg)
"""

from collections import Counter
#Word Count & Keyword for Full Corpus
topics_full = lda_full.show_topics(formatted=False)
flatten_full = [w for w_list in bow_corpus_full for w in w_list]
counter_full = Counter(flatten_full)

topic_weight_full = []
for i, topic in topics_full:
    for word, weight in topic:
        topic_weight_full.append([word, i , weight, counter_full[word]])

data_frame_full = pd.DataFrame(topic_weight_full, columns=['word', 'topic_id', 'importance', 'word_count'])        

fig, axes = plt.subplots(3, 3, figsize=(10,6), sharey=True, dpi=160)

for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=data_frame_full.loc[data_frame_full.topic_id==i, :], color=colors[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=data_frame_full.loc[data_frame_full.topic_id==i, :], color=colors[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=colors[i])
    ax_twin.set_ylim(0, 0.5); ax.set_ylim(0, 100)
    ax.set_title('Topic: ' + str(i+1), color=colors[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(data_frame_full.loc[data_frame_full.topic_id==i, 'word'], rotation=90, horizontalalignment= 'center')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
plt.show()

"""
#Word Count & Keyword for Positive Corpus
topics_pos = lda_pos.show_topics(formatted=False)
flatten_pos = [w for w_list in bow_corpus_pos for w in w_list]
counter_pos = Counter(flatten_pos)

topic_weight_pos = []
for i, topic in topics_pos:
    for word, weight in topic:
        topic_weight_pos.append([word, i , weight, counter_pos[word]])

data_frame_pos = pd.DataFrame(topic_weight_pos, columns=['word', 'topic_id', 'importance', 'word_count'])        

fig, axes = plt.subplots(3, 3, figsize=(10,6), sharey=True, dpi=160)

for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=data_frame_pos.loc[data_frame_pos.topic_id==i, :], color=colors[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=data_frame_pos.loc[data_frame_pos.topic_id==i, :], color=colors[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=colors[i])
    ax_twin.set_ylim(0, 0.5); ax.set_ylim(0, 100)
    ax.set_title('Topic: ' + str(i+1), color=colors[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(data_frame_pos.loc[data_frame_pos.topic_id==i, 'word'], rotation=90, horizontalalignment= 'center')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
plt.show()

#Word Count & Keyword for Negative Corpus
topics_neg = lda_neg.show_topics(formatted=False)
flatten_neg = [w for w_list in bow_corpus_neg for w in w_list]
counter_neg = Counter(flatten_neg)

topic_weight_neg = []
for i, topic in topics_neg:
    for word, weight in topic:
        topic_weight_neg.append([word, i , weight, counter_neg[word]])

data_frame_neg = pd.DataFrame(topic_weight_neg, columns=['word', 'topic_id', 'importance', 'word_count'])        

fig, axes = plt.subplots(3, 3, figsize=(10,6), sharey=True, dpi=160)

for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=data_frame_neg.loc[data_frame_neg.topic_id==i, :], color=colors[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=data_frame_neg.loc[data_frame_neg.topic_id==i, :], color=colors[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=colors[i])
    ax_twin.set_ylim(0, 0.5); ax.set_ylim(0, 100)
    ax.set_title('Topic: ' + str(i+1), color=colors[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(data_frame_neg.loc[data_frame_neg.topic_id==i, 'word'], rotation=90, horizontalalignment= 'center')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
plt.show()
"""

from wordcloud import WordCloud
import matplotlib.colors as mcolors
#Word Cloud Display for Full Corpus
cloud = WordCloud(stopwords=stoplist, background_color='white', width=2500, height=1800, max_words=7, colormap='tab10', color_func=lambda *args, **kwargs: colors[i], prefer_horizontal=1.0)

topics_full = lda_full.show_topics(formatted=False)

fig, axes = plt.subplots(3, 3, figsize=(10, 6))

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words_full = dict(topics_full[i][1])
    cloud.generate_from_frequencies(topic_words_full, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=10))
    plt.gca().axis('off')

plt.axis('off')
plt.tight_layout()
plt.show()

"""
#Word Cloud Display for Positive Corpus
cloud = WordCloud(stopwords=stoplist, background_color='white', width=2500, height=1800, max_words=7, colormap='tab10', color_func=lambda *args, **kwargs: colors[i], prefer_horizontal=1.0)

topics_pos = lda_pos.show_topics(formatted=False)

fig, axes = plt.subplots(3, 3, figsize=(10, 6))

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words_pos = dict(topics_pos[i][1])
    cloud.generate_from_frequencies(topic_words_pos, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=10))
    plt.gca().axis('off')

plt.axis('off')
plt.tight_layout()
plt.show()

#Word Cloud Display for Negative Corpus
cloud = WordCloud(stopwords=stoplist, background_color='white', width=2500, height=1800, max_words=7, colormap='tab10', color_func=lambda *args, **kwargs: colors[i], prefer_horizontal=1.0)

topics_neg = lda_neg.show_topics(formatted=False)

fig, axes = plt.subplots(3, 3, figsize=(10, 6))

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words_neg = dict(topics_neg[i][1])
    cloud.generate_from_frequencies(topic_words_neg, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=10))
    plt.gca().axis('off')

plt.axis('off')
plt.tight_layout()
plt.show()
"""

import pyLDAvis.gensim
import pyLDAvis
import gensim   

#LDA Mallet pyLDAvis for Full Corpus
mallet2lda_full = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_full)
visualizeLDA_full = pyLDAvis.gensim.prepare(mallet2lda_full, bow_corpus_full, dictionary_full)
pyLDAvis.show()

"""
#LDA Mallet pyLDAvis for Postiive Corpus
mallet2lda_pos = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_pos)
visualizeLDA_pos = pyLDAvis.gensim.prepare(mallet2lda_pos, bow_corpus_pos, dictionary_pos)
pyLDAvis.show(visualizeLDA_pos)

#LDA Mallet pyLDAvis for Negative Corpus
mallet2lda_neg = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_neg)
visualizeLDA_neg = pyLDAvis.gensim.prepare(mallet2lda_neg, bow_corpus_neg, dictionary_neg)
pyLDAvis.show(visualizeLDA_neg)
"""