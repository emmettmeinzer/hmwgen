# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:00:27 2020

@author: Emmett
"""
import pickle
import hmw_nonsense_detector_train
import text_summarization_extractive

model_data = pickle.load(open('nonsense_detector.pki', 'rb'))

sensical_questions = []

sentence = text_summarization_extractive.HMW
model_mat = model_data['mat']
#threshold = model_data['thresh']
threshold = 0.0002;
for i in range(len(sentence)):
    if ((hmw_nonsense_detector_train.avg_transition_prob(sentence[i], model_mat) > threshold) == True):
        sensical_questions.append(sentence[i])
    else:
        None
        
file1 = open("Sensical_HMW.txt","w")
for i in range(len(sensical_questions)):
    file1.write("How might we " + sensical_questions[i] + "?")
    file1.write("\n")
    
file2 = open("HMW.txt","w")
for i in range(len(sentence)):
    file2.write("How might we " + sentence[i] + "?")
    file2.write("\n")