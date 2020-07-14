#!/usr/bin/env python
# coding: utf-8

# ## Data Pre-processing

# In[1]:


import os
import gensim.models as models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import random,numpy as np
import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
# from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import operator, random
import sys
import math
import pickle
from nltk.stem import *
from num2words import num2words
from wordcloud import STOPWORDS

def remove_header_footer(final_string):
	new_final_string=""
	flag=1
	tokens=final_string.split('\n\n')
	# Remove tokens[0] and tokens[-1]
	for token in tokens[1:-1]:
		flag+=1
		new_final_string+=str(token)+" "
	flag=0
	return new_final_string

def remove_html(data):
	return BeautifulSoup(data, "html.parser").get_text()

# def remove_btw_sqr(data):
#     fin = re.sub('\[[^]]*\]', '', data)
#     return fin

def fix_contractions(data):
    fin = contractions.fix(data)
    return fin

def words_tokenizer(data):
	words = nltk.word_tokenize(data)
	# tknzr = TweetTokenizer()	
	# tknzr.tokenize(data)
	return words

def remove_non_ascii(words):
	new_words = []
	flag = 0
	for i in range(len(words)):
		flag = 1
		new_word = unicodedata.normalize('NFKD',unicode(words[i]))
		new_word = new_word.encode('ascii','ignore')
		new_word = new_word.decode('utf-8','ignore')
		flag+=1
		new_words.append(new_word)
	return new_words
# def remove_non_ascii(words):
#     new_words = []
#     flag = 0
#     for i in range(len(words)):
#  		flag=1
#  		new_word = unicodedata.normalize('NFKD', words[i]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#  		flag+=1
#  		new_words.append(new_word)
#  	return new_words

def to_lowercase(words):
    new_words = []
    flag = 0
    for i in range(len(words)):
        new_word = words[i].lower()
        flag+=1
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    flag = 0
    for i in range(len(words)):
    	flag+=1
        new_word = re.sub(r'([^\w\s])|_+', '', words[i])
        if new_word != '':
        	flag=0
        	new_words.append(new_word)
    return new_words

# def replace_numbers(words):
#     p = inflect.engine()
#     new_words = []
#     flag = 0
#     for i in range(len(words)):
#     	flag = 1
#         if words[i].isdigit():
#             new_word = p.number_to_words(words[i])
#             flag+=1
#             new_words.append(new_word)
#         else:
#         	flag = 0
#         	new_words.append(words[i])
#     return new_words

def replace_numbers(words):
	new_words = []
	for i in range(len(words)):
		if words[i].isdigit():
			temp_word = num2words(words[i])
			new_words.append(temp_word)
		else:
			new_words.append(words[i])
	return new_words

def remove_stopwords(words):
    new_words = []
    flag = 0
    for i in range(len(words)):
    	flag = 1
        if words[i] not in stopwords.words('english') and words[i] not in STOPWORDS:
        	flag+=1
        	new_words.append(words[i])
    return new_words

def remove_numbers(words):
    new_words = []
    flag=0
    for i in range(len(words)):
        if words[i].isdigit():
            flag+=1
        else:
            new_words.append(words[i])
    return new_words

def stemming(words):
	new_words = []
	stemmer = PorterStemmer()
	for i in range(len(words)):
		new_words.append(stemmer.stem(words[i]))
	return new_words

def preprocess_input_sentence(data):
	# data = remove_header_footer(data)
	data = remove_html(data)
	# data = remove_btw_sqr(data)
	data = fix_contractions(data)
	words = words_tokenizer(data)
	words = remove_non_ascii(words)
	words = to_lowercase(words)
	words = remove_punctuation(words)
	words = replace_numbers(words)
	words = stemming(words)
	words = remove_stopwords(words)
	return words

file_mapping_count = -1
prepro_data_dic = {}
count_to_name = {}
name_to_count = {}
file_titles = {}
toremove = []
ground_labels = []

with open('prepro_files.pkl') as f:
	prepro_data_dic = pickle.load(f)

# with open('name_to_count.pkl') as f:
# 	name_to_count = pickle.load(f)

# with open('count_to_name.pkl') as f:
# 	count_to_name = pickle.load(f)

with open('ground_labels.pkl') as f:
	ground_labels = pickle.load(f)

# ground_count = -1
# for i in os.listdir('20_newsgroups/'):
#     ground_count+=1
#     for j in sorted(os.listdir('20_newsgroups/'+i)):
#         file_mapping_count+=1
# #         print file_mapping_count
#         file_name = i+'/'+j
# #         print file_name
#         count_to_name[file_mapping_count] = file_name
#         name_to_count[file_name] = file_mapping_count
#         file_name_path = '20_newsgroups/'+i+'/'+j
#         temp_data = open(file_name_path,'rb').read().decode('utf-8', 'ignore').lower()
#         prepro_data = preprocess_input_sentence(temp_data)
#         prepro_data_dic[file_mapping_count] = prepro_data
#         ground_labels.append(ground_count)


# ## Data Splitting Function

# In[2]:


import random

def data_splitting(data,labels,ratio,seed):
    
    classwise_data = {}
    for i in range(len(labels)):
        if labels[i] not in classwise_data:
            classwise_data[labels[i]] = []
        classwise_data[labels[i]].append(data[i])
    
    final_train_data = []
    final_test_data = []
    final_train_labels = []
    final_test_labels = []
    final_train_docids = []
    final_test_docids = []
    
    for k,v in classwise_data.iteritems():
        shuf_ind = []
        for i in range(len(v)):
            shuf_ind.append(i)
        random.Random(seed).shuffle(shuf_ind)
        
        lim = int(len(shuf_ind)*ratio)
        train_ind = shuf_ind[:lim]
        test_ind = shuf_ind[lim:]

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for i in range(len(train_ind)):
            train_data.append(classwise_data[k][train_ind[i]])
            train_labels.append(k)
        for i in range(len(test_ind)):
            test_data.append(classwise_data[k][test_ind[i]])
            test_labels.append(k)
        final_train_data+=train_data
        final_test_data+=test_data
        final_train_labels+=train_labels
        final_test_labels+=test_labels
        
    return final_train_data,final_train_labels,final_test_data,final_test_labels


# ## Fucntions to check different data stats
# 

# In[3]:


# train_data,train_labels,test_data,test_labels = data_splitting(prepro_data_dic,ground_labels,0.7)

classwise_vocab = {}

for i in range(len(prepro_data_dic)):
#         term = prepro_data_dic[i]
        label = ground_labels[i]
        if label not in classwise_vocab:
            classwise_vocab[label] = {}
        for j in range(len(prepro_data_dic[i])):
            term = prepro_data_dic[i][j]
            if term not in classwise_vocab[label]:
                classwise_vocab[label][term] = 0
            classwise_vocab[label][term]+=1
            


# In[4]:


for k,v in classwise_vocab.iteritems():
    print k, len(v)


# In[5]:


# Creating vocabulary
vocab = set()
for k,v in classwise_vocab.iteritems():
    for key, val in v.iteritems():
        vocab.add(key)
vocab = list(vocab)


# In[6]:


len(vocab)


# # Part 1: Navie Bayes

# In[7]:


import numpy as np
import pandas as pd

# ----------- Testing for different ratio valuex ------------------

ratios = [0.5,0.7,0.8,0.9]

for rat in range(len(ratios)):
    ratio = ratios[rat]
    train_data,train_labels,test_data,test_labels = data_splitting(prepro_data_dic,ground_labels,ratio,6)
    print "Ratio: "+str(ratios[rat])
    train_classwise_vocab = {}
    train_vocab = set()
    for i in range(len(train_data)):
        
        class_val = train_labels[i]
        if class_val not in train_classwise_vocab:
            train_classwise_vocab[class_val] = {}
        
        for j in range(len(train_data[i])):
            
            term = train_data[i][j]
            train_vocab.add(term)
            
            if term not in train_classwise_vocab[class_val]: 
                train_classwise_vocab[class_val][term] = 0
            train_classwise_vocab[class_val][term]+=1
            
    train_vocab = list(train_vocab)
    
#     print "Train vocab built."
    
    # ---------------- Training -----------------------------
    
    # prior probabilities

    train_prior = {}
    for i in range(len(train_labels)):
        temp_class = train_labels[i]
        if temp_class not in train_prior:
            train_prior[temp_class] = 0
        train_prior[temp_class]+=1

    prior = {}
    for k,v in train_prior.iteritems():
        prior[k] = np.log((train_prior[k]*1.0)/(len(train_data)))
    
#     print "Priors generated."
    
    cond_prob = {}
    for i in range(len(train_classwise_vocab)):
        
        class_val = i
        class_terms = train_classwise_vocab[class_val]
        
        tsum = 0
        for k,v in class_terms.iteritems():
            tsum+=v
        
        for k,v in class_terms.iteritems():
            if class_val not in cond_prob:
                cond_prob[class_val] = {}
            cond_prob[class_val][k] = np.log(((v*1.0)+1)/(tsum+len(train_vocab)))
    
#     print "Train conditional probabilities generated."

    # ----------------- Testing --------------------- 

    test_docs = {}

    for i in range(len(test_data)):
        if test_labels[i] not in test_docs:
            test_docs[test_labels[i]] = []
        test_docs[test_labels[i]].append(test_data[i])

    classwise_sum = {}
    for k,v in classwise_vocab.iteritems():
        if k not in classwise_sum:
            classwise_sum[k] = 0
        for key,val in v.iteritems():
            classwise_sum[k]+=val
    
    cor_count=0
    true_class = []
    predicted_class = []
    for i in range(len(test_data)):
        ground_label = test_labels[i]
        doc_val = test_data[i]
        score = {}
        max_val = -float("inf")
        max_class = -1
        for j in range(5):
            score[j] = 0
            score[j]+=prior[j]
            for term in doc_val:
                if term not in cond_prob[j]:
                    score[j]+=np.log(1.0/len(train_vocab))*100
                else:
                    score[j]+=cond_prob[j][term]
                    
            if score[j]>max_val:
                max_val=score[j]
                max_class=j
        if max_class == ground_label:
            cor_count+=1
        predicted_class.append(max_class)
        true_class.append(ground_label)
    
    print "Accuracy: " + str(((cor_count*1.0)/len(test_data))*100) + "%"

    # -------------- Confusion Matrix ---------------------

    mat = np.zeros((len(train_classwise_vocab), len(train_classwise_vocab)))
    flag=0
    for it in range(len(predicted_class)):
        flag+=1
        mat[int(true_class[it])][int(predicted_class[it])] += 1

    df = pd.DataFrame(mat)
    print("True vs Predicted")
    flag+=1
    print(mat)
    print ""


# # Part 2: TF-IDF scoring for feature selection

# In[8]:


import numpy as np
import pandas as pd

ratio = 0.7
train_data,train_labels,test_data,test_labels = data_splitting(prepro_data_dic,ground_labels,ratio,7)


# In[9]:


train_classwise_vocab = {}
train_vocab = set()
for i in range(len(train_data)):

    class_val = train_labels[i]
    if class_val not in train_classwise_vocab:
        train_classwise_vocab[class_val] = {}

    for j in range(len(train_data[i])):

        term = train_data[i][j]
        train_vocab.add(term)

        if term not in train_classwise_vocab[class_val]: 
            train_classwise_vocab[class_val][term] = 0
        train_classwise_vocab[class_val][term]+=1

train_vocab = list(train_vocab)

train_classwise_sum = {}

for k,v in train_classwise_vocab.iteritems():
    if k not in train_classwise_sum:
        train_classwise_sum[k] = 0
    for key,val in v.iteritems():
        train_classwise_sum[k]+=val


# ## TF-IDF calculation

# In[10]:


# ---------------- TERM FREQUENCY(TF) --------------------

# TF = {}

# train_data_freq = {}
# for i in range(len(train_data)):
#     for j in range(len(train_data[i])):
#         term = train_data[i][j]
#         if i not in train_data_freq:
#             train_data_freq[i] = {}
#         if term not in train_data_freq[i]:
#             train_data_freq[i][term] = 0
#         train_data_freq[i][term]+=1

# for i in range(len(train_vocab)):
#     term = train_vocab[i]
#     for j in range(len(train_data)):
#         try:
#             tfval = (train_data_freq[j][term]*1.0)/(len(train_data[j]))
#             if term not in TF:
#                 TF[term] = 0
#             TF[term]+=tfval
#         except:
#             pass

# with open('TF_short.pkl','wb') as f:
#     f.write(pickle.dumps(TF))

# with open('TF_short.pkl') as f:
#     TF = pickle.load(f)
    
# --------------------- DOCUMENT FREQUENCY(DF) -----------------------------

# df_dic = {}

# for i in range(len(train_vocab)):
#     termval = train_vocab[i]
#     for j in range(len(train_data)):
#         docval = train_data[j]
#         if termval in docval:
#             if termval not in df_dic:
#                 df_dic[termval] = []
#             df_dic[termval].append(1)

# with open('DF_short.pkl','wb') as f:
#     f.write(pickle.dumps(df_dic))

# with open('DF_short.pkl') as f:
#     DF_classwise = pickle.load(f)

# ------------------------ TF-IDF ----------------------------

# TFIDF = {}
# for i in range(len(train_vocab)):
#     term = train_vocab[i]
#     tf_value = TF[term]
#     df_value = len(df_dic[term])
#     body_score = (1+tf_value)*(math.log10(len(train_data)/((1+df_value)*1.0)))
#     TFIDF[term] = body_score

# with open('TFIDF_short.pkl') as f:
#     TFIDF = pickle.load(f)

# ------------------- TOP-K FEATURE SELECTION ----------------------------
# import operator
# sorted_sim_results = sorted(TFIDF.items(), key=operator.itemgetter(1),reverse=True)

# top_ind = len(sorted_sim_results)*60/100
# topk = sorted_sim_results[:top_ind]
# topk_dic = {}

# for i in range(len(topk)):
#     topk_dic[topk[i][0]] = topk[i][1]

# with open('topk_tfidf.pkl','wb') as f:
#     f.write(pickle.dumps(topk_dic))

topk_dic = {}
with open('topk_tfidf.pkl') as f:
    topk_dic = pickle.load(f)


# ## Naive Bayes

# In[11]:


# ---------------- Training -----------------------------
print "Ratio: " + str(ratio)
# prior probabilities

train_prior = {}
for i in range(len(train_labels)):
    temp_class = train_labels[i]
    if temp_class not in train_prior:
        train_prior[temp_class] = 0
    train_prior[temp_class]+=1

prior = {}
for k,v in train_prior.iteritems():
    prior[k] = np.log((train_prior[k]*1.0)/(len(train_data)))

# print "Priors generated."

cond_prob = {}
for i in range(len(train_classwise_vocab)):

    class_val = i
    class_terms = train_classwise_vocab[class_val]

    tsum = 0
    for k,v in class_terms.iteritems():
        tsum+=v

    for k,v in class_terms.iteritems():
        if class_val not in cond_prob:
            cond_prob[class_val] = {}
        cond_prob[class_val][k] = np.log(((v*1.0)+1)/(tsum+len(train_vocab)))

# print "Train conditional probabilities generated."

# ----------------- Testing --------------------- 

test_docs = {}

for i in range(len(test_data)):
    if test_labels[i] not in test_docs:
        test_docs[test_labels[i]] = []
    test_docs[test_labels[i]].append(test_data[i])

classwise_sum = {}
for k,v in classwise_vocab.iteritems():
    if k not in classwise_sum:
        classwise_sum[k] = 0
    for key,val in v.iteritems():
        classwise_sum[k]+=val

cor_count=0
true_class = []
predicted_class = []
for i in range(len(test_data)):
    ground_label = test_labels[i]
    doc_val = test_data[i]
    score = {}
    max_val = -float("inf")
    max_class = -1
    for j in range(5):
        score[j] = 0
        score[j]+=prior[j]
        for term in doc_val:
            if (term in topk_dic) and (term in cond_prob[j]):
                score[j]+=cond_prob[j][term]
            else:
                score[j]+=np.log(1.0/len(train_vocab))*100
        if score[j]>max_val:
            max_val=score[j]
            max_class=j
    if max_class == ground_label:
        cor_count+=1
    predicted_class.append(max_class)
    true_class.append(ground_label)

# print cor_count
print "Accuracy: " + str(((cor_count*1.0)/len(test_data))*100) + "%"

# -------------- Confusion Matrix ---------------------

mat = np.zeros((len(train_classwise_vocab), len(train_classwise_vocab)))
flag=0
for it in range(len(predicted_class)):
    flag+=1
    mat[int(true_class[it])][int(predicted_class[it])] += 1

df = pd.DataFrame(mat)
print("True vs Predicted")
flag+=1
print(mat)
# print len(train_vocab)
print ""

