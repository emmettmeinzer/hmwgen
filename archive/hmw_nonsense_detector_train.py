# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 22:56:11 2020

@author: Emmett
"""
import re
import math
import pickle

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

def train():
    """ Write a simple model as a pickle file """
    k = len(accepted_words)
    # Assume we have seen 10 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    counts = [[10 for i in range(k)] for i in range(k)]

    # Count transitions from full text file (RedditSamples.csv)
    for line in open('full.txt'):
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

    # Find the probability of generating a few arbitrarily choosen good and
    # bad phrases.
    good_probs = [avg_transition_prob(l, counts) for l in open('good.txt')]
    bad_probs = [avg_transition_prob(l, counts) for l in open('bad.txt')]

    # Assert that we actually are capable of detecting false sentences.
    assert min(good_probs) > max(bad_probs)

    # And pick a threshold halfway between the worst good and best bad HMW questions.
    thresh = (min(good_probs) + max(bad_probs)) / 2
    pickle.dump({'mat': counts, 'thresh': thresh}, open('nonsense_detector.pki', 'wb'))

def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]] #find the probability distribution of pos[a] occuring next to pos[b]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))

if __name__ == '__main__':
    train()