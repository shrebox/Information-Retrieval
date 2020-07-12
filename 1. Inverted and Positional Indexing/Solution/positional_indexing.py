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
#         	new_words.append(word)
#     return new_words

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
	data = remove_header_footer(data)
	data = remove_html(data)
	# data = remove_btw_sqr(data)
	data = fix_contractions(data)
	words = words_tokenizer(data)
	words = remove_non_ascii(words)
	words = to_lowercase(words)
	words = remove_punctuation(words)
	# words = replace_numbers(words)
	# words = remove_stopwords(words)
	words = stemming(words)
	return words

index_to_file_mappings = {}
file_to_index_mappings = {}
phrasal_inverted_index = {}
file_mapping_count = 0

# # ---------------------------- Phrasal indexing: data processing and inverted index creation ---------------------------------

# for i in os.listdir('20_newsgroups/'):
# 	corpus = []
# 	for j in sorted(os.listdir('20_newsgroups/'+i)):
# 		file_mapping_count+=1
# 		file_name = i+'/'+j
# 		index_to_file_mappings[file_mapping_count] = file_name
# 		file_to_index_mappings[file_name] = file_mapping_count
# 		file_name_path = '20_newsgroups/'+i+'/'+j
# 		temp_data = open(file_name_path,'rb').read().decode('utf-8', 'ignore').lower()
# 		prepro_data = preprocess_input_sentence(temp_data)
# 		for k in range(len(prepro_data)):
# 			if prepro_data[k] not in phrasal_inverted_index:
# 				phrasal_inverted_index[prepro_data[k]] = []
# 				tlist = {}
# 				phrasal_inverted_index[prepro_data[k]].append(0)
# 				phrasal_inverted_index[prepro_data[k]].append(tlist)
# 			phrasal_inverted_index[prepro_data[k]][0]+=1
# 			if file_mapping_count not in phrasal_inverted_index[prepro_data[k]][1]: 
# 				phrasal_inverted_index[prepro_data[k]][1][file_mapping_count] = []
# 			phrasal_inverted_index[prepro_data[k]][1][file_mapping_count].append(k)

# ------------------------- loading stored inverted indexes and filename mappings ------------------------------

with open('phrasal_inverted_index_with_stopwords_q3.pkl') as f:
	phrasal_inverted_index = pickle.load(f)

with open('file_to_index_mappings_with_stopwords_q3.pkl') as f:
	file_to_index_mappings = pickle.load(f)

with open('index_to_file_mappings_with_stopwords_q3.pkl') as f:
	index_to_file_mappings = pickle.load(f)

# ---------------------------------- Positional Index Search -------------------------------------

def merging_documents(doc1,doc2):
	
	final = {}
	x = doc1
	y = doc2
	common_dics = sorted(list(set(x) & set(y)))

	for i in range(len(common_dics)):
		indocs1 = x[common_dics[i]]
		indocs2 = y[common_dics[i]]
		ind1 = ind2 = 0
		while(ind2<len(indocs2) and ind1<len(indocs1)):
			differ = indocs2[ind2]-indocs1[ind1]
			if differ > 1:
				ind1+=1
			elif differ < 1:
				ind2+=1
			else:
				if common_dics[i] not in final:
					final[common_dics[i]] = []
				final[common_dics[i]].append(indocs2[ind2])
				ind1+=1
				ind2+=1

	return final

while True:
	try:
		print ""
		inp_phrase = str(raw_input("Enter the phrase (-1 to exit): ")).lower().split()

		if inp_phrase[0]!="-1":
			stemmer = PorterStemmer()

			query_len = len(inp_phrase)

			if query_len == 0:
				print "No query entered!"
			elif query_len ==1:
				tdocs = phrasal_inverted_index[stemmer.stem(inp_phrase[0])][1]
				for k,v in tdocs.iteritems():
					print index_to_file_mappings[k]
			else:
				ext = merging_documents(phrasal_inverted_index[stemmer.stem(inp_phrase[0])][1],phrasal_inverted_index[stemmer.stem(inp_phrase[1])][1])

				for i in range(2,len(inp_phrase)):
					ext = merging_documents(ext,phrasal_inverted_index[stemmer.stem(inp_phrase[i])][1])

				for k, v in ext.items():
				    print(index_to_file_mappings[k], v)
			print len(ext)
		else:
			break
	except:
		pass







