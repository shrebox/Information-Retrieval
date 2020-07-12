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
	words = remove_stopwords(words)
	words = stemming(words)
	return words

file_mappings = {}
inverted_index = {}
# file_mapping_count = 0

# ---------------------------- data processing and inverted index creation ---------------------------------

# for i in os.listdir('20_newsgroups/'):
# 	corpus = []
# 	for j in sorted(os.listdir('20_newsgroups/'+i)):
# 		file_mapping_count+=1
# 		file_name = i+'/'+j
# 		file_mappings[file_mapping_count] = file_name
# 		file_name_path = '20_newsgroups/'+i+'/'+j
# 		temp_data = open(file_name_path,'rb').read().decode('utf-8', 'ignore').lower()
# 		prepro_data = preprocess_input_sentence(temp_data)
# 		prepro_data = list(set(prepro_data))
# 		for k in range(len(prepro_data)):
# 			if prepro_data[k] not in inverted_index:
# 				inverted_index[prepro_data[k]] = []
# 				tlist = []
# 				inverted_index[prepro_data[k]].append(0)
# 				inverted_index[prepro_data[k]].append(tlist)
# 			inverted_index[prepro_data[k]][0]+=1
# 			inverted_index[prepro_data[k]][1].append(file_mapping_count)

# with open('inverted_index.pkl','wb') as f:
# 	f.write(pickle.dumps(inverted_index))

# with open('filename_mappings.pkl','wb') as f:
# 	f.write(pickle.dumps(file_mappings))

# ------------------------- loading stored inverted indexes and filename mappings ------------------------------

with open('inverted_index.pkl') as f:
	inverted_index = pickle.load(f)

with open('filename_mappings.pkl') as f:
	file_mappings = pickle.load(f)

# ---------------------------------- Part 2: 'or', 'and', and 'not' commands -------------------------------------

def or_command(x,y):
	return list(set(x) | set(y))

def and_command(x,y):
	return list(set(x) & set(y))

def not_command(x):
	data = []
	total_data = file_mappings.keys()
	# try:
	x_docs = x
	for i in range(len(total_data)):
		if total_data[i] not in x_docs:
			data.append(total_data[i])
	# except Exception, e:
	# 	print e
	return data

stemmer = PorterStemmer()
flag = 1
while(flag==1):
	try:
		print "1. x OR y \n2. x AND y \n3. x AND NOT y \n4. x OR NOT y \n5. Exit \n"
		inp = int(raw_input("Choose option: "))
		print ""
		final = []
		if inp == 1:
			x_value = stemmer.stem(str(raw_input("Enter x value: ")).lower())
			y_value = stemmer.stem(str(raw_input("Enter y value: ")).lower())
			try:
				x_ivt = inverted_index[x_value][1]
				y_ivt = inverted_index[y_value][1]
				final = or_command(x_ivt,y_ivt)
			except Exception,e:
				print e
		elif inp == 2:
			x_value = stemmer.stem(str(raw_input("Enter x value: ")).lower())
			y_value = stemmer.stem(str(raw_input("Enter y value: ")).lower())
			try:
				x_ivt = inverted_index[x_value][1]
				y_ivt = inverted_index[y_value][1]
				final = and_command(x_ivt,y_ivt)
			except Exception,e:
				print e
		elif inp == 3:
			x_value = stemmer.stem(str(raw_input("Enter x value: ")).lower())
			y_value = stemmer.stem(str(raw_input("Enter y value: ")).lower())
			try:
				y_ivt = inverted_index[y_value][1]
				not_y_value = not_command(y_ivt)
				x_ivt = inverted_index[x_value][1]
				final = and_command(x_ivt,not_y_value)
			except Exception,e:
				print e
		elif inp == 4:
			x_value = stemmer.stem(str(raw_input("Enter x value: ")).lower())
			y_value = stemmer.stem(str(raw_input("Enter y value: ")).lower())
			try:
				y_ivt = inverted_index[y_value][1]
				not_y_value = not_command(y_ivt)
				x_ivt = inverted_index[x_value][1]
				final = or_command(x_ivt,not_y_value)
			except Exception,e:
				print e
		elif inp==5:
			flag=0
			break
		else:
			print "Enter valid option!"

		print ""
		for i in range(len(final)):
			print file_mappings[final[i]]
		print len(final)
		print ""
	except Exception,e:
		pass






