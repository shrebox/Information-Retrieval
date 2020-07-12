#!/usr/bin/env python
# coding: utf-8

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
#       flag=1
#       new_word = unicodedata.normalize('NFKD', words[i]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#       flag+=1
#       new_words.append(new_word)
#   return new_words

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
#       flag = 1
#         if words[i].isdigit():
#             new_word = p.number_to_words(words[i])
#             flag+=1
#             new_words.append(new_word)
#         else:
#           flag = 0
#           new_words.append(words[i])
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
        if words[i] not in stopwords.words('english'):
            flag+=1
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

file_mapping_count = 0
prepro_data_dic = {}
count_to_name = {}
name_to_count = {}
file_titles = {}
toremove = []

with open('prepro_files.pkl') as f:
    prepro_data_dic = pickle.load(f)

with open('name_to_count.pkl') as f:
    name_to_count = pickle.load(f)

with open('count_to_name.pkl') as f:
    count_to_name = pickle.load(f)

with open('toremove.pkl') as f:
    toremove = pickle.load(f)

for i in range(len(toremove)):
    del prepro_data_dic[toremove[i]]

with open('count_to_titles.pkl') as f:
    file_titles = pickle.load(f)


# ----------------------file titles -----------------------------
# file_titles = []

# with open('file_titles.txt') as f:
#   for line in f:
#       file_titles.append(line)

# for i in range(len(file_titles)):
#   if i%2==0:
#       name = file_titles[i].split('\t')[0]
#       size = file_titles[i].split('\t')[1]
#       if name not in titles:
#           titles[name] = {}
#       titles[name]['size'] = int(size.rstrip('\n'))
#       # print titles
#   else:
#       titles[name]['title'] = file_titles[i]

# -----------------------------reading and processing files ----------------------
# for i in os.listdir('stories/'):
#   try:
#       file_name_path = 'stories/'+i
#       temp_data = open(file_name_path,'rb').read().decode('utf-8', 'ignore').lower()
#       file_mapping_count+=1
#       file_name = i
#       count_to_name[file_mapping_count] = file_name
#       name_to_count[file_name] = file_mapping_count
#       if (".html" in file_name or ".descs" in file_name or ".header" in file_name or ".footer" in file_name or ".musings" in file_name):
#           toremove.append(file_mapping_count)
#       prepro_data = preprocess_input_sentence(temp_data)
#       prepro_data_dic[file_mapping_count] = prepro_data
#   except:
#       pass

# for i in os.listdir('stories/'):
#   try:
#       for j in sorted(os.listdir('stories/'+i)):
#           file_name_path = 'stories/'+i+'/'+j
#           temp_data = open(file_name_path,'rb').read().decode('utf-8', 'ignore').lower()
#           file_mapping_count+=1
#           file_name = i+'/'+j
#           count_to_name[file_mapping_count] = file_name
#           name_to_count[file_name] = file_mapping_count
#           if (".html" in file_name or ".descs" in file_name or ".header" in file_name or ".footer" in file_name or ".musings" in file_name):
#               toremove.append(file_mapping_count)
#           prepro_data = preprocess_input_sentence(temp_data)
#           prepro_data_dic[file_mapping_count] = prepro_data
#   except:
#       pass


# In[2]:


# ------------------------------------- creating dictionaries -----------------------------------

vocab = []

# for k,v in prepro_data_dic.iteritems():
#     for i in range(len(v)):
#         if v[i] not in vocab:
#             vocab.append(v[i])

# with open('vocab.pkl','wb') as f:
#     f.write(pickle.dumps(vocab))

with open('vocab.pkl') as f:
    vocab = pickle.load(f)


# In[3]:


len(vocab)


# In[4]:


# TF Dictionary

tf_dic = {}

# for k,v in prepro_data_dic.iteritems():
#     tf_dic[k] = {}
#     for i in range(len(v)):
#         if v[i] not in tf_dic[k]:
#             tf_dic[k][v[i]] = 0
#         tf_dic[k][v[i]] += 1

# with open('tf_dic.pkl','wb') as f:
#     f.write(pickle.dumps(tf_dic))

with open('tf_dic.pkl') as f:
    tf_dic = pickle.load(f)


# In[5]:


len(tf_dic)


# In[6]:


# DF Dictionary

df_dic = {}

# for i in range(len(vocab)):
#         for k,v in prepro_data_dic.iteritems():
#             if vocab[i] in v:
#                 if vocab[i] not in df_dic:
#                     df_dic[vocab[i]] = []
#                 df_dic[vocab[i]].append(k)

# with open('df_dic.pkl','wb') as f:
#     f.write(pickle.dumps(df_dic))

with open('df_dic.pkl') as f:
    df_dic = pickle.load(f)


# In[7]:


len(df_dic)


# In[8]:


prepro_file_titles = {}

for k,v in file_titles.iteritems():
    temp = preprocess_input_sentence(v['title'])
    prepro_file_titles[k] = temp

for k,v in prepro_file_titles.iteritems():
    print k,v
    break


# In[9]:


# Title document 

tf_tit_dic = {}
# for k,v in prepro_file_titles.iteritems():
#     tf_tit_dic[k] = {}
#     for i in range(len(v)):
#         if v[i] not in tf_tit_dic[k]:
#             tf_tit_dic[k][v[i]] = 0
#         tf_tit_dic[k][v[i]] += 1

# with open('tf_tit_dic.pkl','wb') as f:
#     f.write(pickle.dumps(tf_tit_dic))


with open('tf_tit_dic.pkl') as f:
    tf_tit_dic = pickle.load(f)


# In[10]:


df_tit_dic = {}

# for i in range(len(vocab)):
#         for k,v in prepro_file_titles.iteritems():
#             if vocab[i] in v:
#                 if vocab[i] not in df_tit_dic:
#                     df_tit_dic[vocab[i]] = []
#                 df_tit_dic[vocab[i]].append(k)

# with open('df_tit_dic.pkl','wb') as f:
#     f.write(pickle.dumps(df_tit_dic))

with open('df_tit_dic.pkl') as f:
    df_tit_dic = pickle.load(f) 


# In[26]:

# In[13]:


# # Part 2: Tf-Idf based vector space document retrieval:

tfresults = {}
# query = vocab
# for i in range(len(query)):
#     term = query[i]
#     for k,v in prepro_data_dic.iteritems():
#         tf_value = 0
#         if term in tf_dic[k]:
#             tf_value = tf_dic[k][term]/(len(prepro_data_dic[k])*1.0)
#         df_value = 0
#         if term in df_dic:
#             df_value = len(df_dic[term])
#         body_score = (1+tf_value)*(math.log10(len(prepro_data_dic)/((1+df_value)*1.0)))
#         tf_tit_value = 0
#         if term in tf_tit_dic[k]:
#             tf_tit_value = tf_tit_dic[k][term]/(len(prepro_file_titles[k])*1.0)
#         df_tit_value = 0
#         if term in df_tit_dic:
#             df_tit_value = len(df_tit_dic[term])
#         title_score = (1+tf_tit_value)*(math.log10(len(prepro_file_titles)/((1+df_tit_value)*1.0)))
#         total_score = (0.6*(title_score))+(0.4*(body_score))
#         if k not in tfresults:
#             tfresults[k] = []
#         tfresults[k].append(total_score)
# #         tfresults[k].append((term,total_score))

# with open('document_term_tfidf_without_title.pkl','wb') as f:
#     f.write(pickle.dumps(tfresults))

with open('document_term_tfidf.pkl') as f:
    tfresults = pickle.load(f)


# In[ ]:



# query

flag=1

while(flag==1):

    chos = int(raw_input("Enter the choice of method 1(non-vector based) or 2(vector based) [Exit = -1]: "))

    if chos==1:

        import math

        query = str(raw_input('Enter query: '))
        query = preprocess_input_sentence(query)

        print query
        results = {}
        for i in range(len(query)):
            term = query[i]
            for k,v in prepro_data_dic.iteritems():
                tf_value = 0
                if term in tf_dic[k]:
                    tf_value = tf_dic[k][term]/(len(prepro_data_dic[k])*1.0)
                df_value = 0
                if term in df_dic:
                    df_value = len(df_dic[term])
                body_score = (1+tf_value)*(math.log10(len(prepro_data_dic)/((1+df_value)*1.0)))
                tf_tit_value = 0
                if term in tf_tit_dic[k]:
                    tf_tit_value = tf_tit_dic[k][term]/(len(prepro_file_titles[k])*1.0)
                df_tit_value = 0
                if term in df_tit_dic:
                    df_tit_value = len(df_tit_dic[term])
                title_score = (1+tf_tit_value)*(math.log10(len(prepro_file_titles)/((1+df_tit_value)*1.0)))
                total_score = (0.6*(title_score))+(0.4*(body_score))
                if term not in results:
                    results[term] = {}
                results[term][k] = total_score


        # In[27]:


        k_value = int(raw_input("Enter k for top k documents: "))

        combine_results = {}
        for k,v in results.iteritems():
            for key, value in v.iteritems():
                if key not in combine_results:
                    combine_results[key] = 0
                combine_results[key]+=value

        sorted_results = sorted(combine_results.items(), key=operator.itemgetter(1),reverse=True)
        # print sorted_results
        sorted_results = sorted_results[:k_value]

        print ""
        for i in range(len(sorted_results)):
            print str(count_to_name[sorted_results[i][0]])+"\t"+str(sorted_results[i][1])

    elif chos ==2:


        query = str(raw_input('Enter query: '))
        query = preprocess_input_sentence(query)

        query_frequency = {}

        for i in range(len(query)):
            if query[i] not in query_frequency:
                query_frequency[query[i]]=0
            query_frequency[query[i]]+=1

        query_vector = []
        for i in range(len(vocab)):
            if vocab[i] in query_frequency:
                query_vector.append(query_frequency[vocab[i]])
            else:
                query_vector.append(0)


        # In[29]:


        len(query_vector)


        # In[30]:


        # Cosine similarity

        import numpy as np

        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        cosine_sim_results = {}

        query_vector = np.array(query_vector)

        for k,v in tfresults.iteritems():
            doc_vector = np.array(v)
            simval = cos_sim(query_vector,doc_vector)
            cosine_sim_results[k] = simval


        # In[31]:


        len(cosine_sim_results)


        # In[32]:


        k_value = int(raw_input("Enter k for top k documents: "))

        sorted_sim_results = sorted(cosine_sim_results.items(), key=operator.itemgetter(1),reverse=True)
        sorted_sim_results = sorted_sim_results[:k_value]

        print ""
        for i in range(len(sorted_sim_results)):
            print str(count_to_name[sorted_sim_results[i][0]])+"\t"+str(sorted_sim_results[i][1])


        # In[ ]:

    elif chos == -1:
        flag=0
        
    else:
        print "Please enter correct option!"



