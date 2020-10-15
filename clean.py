# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:39:05 2019

@author: Emmett
"""

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
    temp3 = strip_multiple_whitespaces(temp2)
    final = stem_text(temp3)
    corpus.append(final)
    sampNum += 1

#print corpus

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

#print(corpus_removeOne)

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

#print(bow_corpus)

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
lda.print_topics(9)

for doc in corpus_LDA:
    if doc == []:
        None
    else:
        print(doc)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

n_topics = 9
topic_weights = []
for row_list in lda[bow_corpus]:
    tmp = np.zeros(n_topics)
    for i, w in row_list:
        tmp[i] = w
    topic_weights.append(tmp)
    
arr = pd.DataFrame(topic_weights).fillna(0).values

topic_num = np.argmax(arr, axis=1)

tsne_model = TSNE(n_components=3, random_state=None, method='barnes_hut', 
                  angle=0.5, init='pca')

tsne_lda = tsne_model.fit_transform(arr)

mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
"""
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.xlabel('t-SNE1'.translate(sub))
plt.ylabel('t-SNE2'.translate(sub))
plt.title('t-SNE Plot of Topics within Air Travel Human Sentiment Corpus')

plt.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])

count = 0
for i in topic_num:
    plt.annotate(i, (tsne_lda[count,0],tsne_lda[count,1]))
    count += 1
    
plt.show()
"""
###

import pyLDAvis.gensim
import pyLDAvis

visualizeLDA = pyLDAvis.gensim.prepare(lda, bow_corpus, dictionary)

pyLDAvis.show(visualizeLDA)

###

from collections import Counter

topics = lda.show_topics(formatted=False)
flatten = [w for w_list in corpus_removeOne for w in w_list]
counter = Counter(flatten)

topic_weight = []
for i, topic in topics:
    for word, weight in topic:
        topic_weight.append([word, i , weight, counter[word]])

data_frame = pd.DataFrame(topic_weight, columns=['word', 'topic_id', 'importance', 'word_count'])        

fig, axes = plt.subplots(3, 3, figsize=(10,6), sharey=True, dpi=160)

for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=data_frame.loc[data_frame.topic_id==i, :], color=mycolors[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=data_frame.loc[data_frame.topic_id==i, :], color=mycolors[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=mycolors[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 250)
    ax.set_title('Topic: ' + str(i+1), color=mycolors[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(data_frame.loc[data_frame.topic_id==i, 'word'], rotation=90, horizontalalignment= 'center')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
plt.show()