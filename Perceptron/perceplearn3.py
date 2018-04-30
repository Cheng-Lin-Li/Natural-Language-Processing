#!/usr/local/bin/python3.6
# encoding: utf-8
'''
Perceptron.perceplearn -- This perceptron classifiers learner program include vanilla and averaged models 


Perceptron.perceplearn is a perceptron classifiers learner program which include vanilla and averaged models 
to identify hotel reviews as either true or fake, and either positive or negative. 
The word tokens will be treated as features, or other features may devise from the text. 
The learner will store weights and bias into model file and pass to classifier to perform classification tasks.

The argument is a single file containing the training data; the program will learn perceptron models, and write 
the model parameters to two files: 
    1. vanillamodel.txt for the vanilla perceptron, and 
    2. averagedmodel.txt for the averaged perceptron. 

It defines classes_and_methods

@author:     Cheng-Lin Li a.k.a. Clark Li@University of Southern California 2018. All rights reserved.

@copyright:  2018 organization_name. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    chenglil@usc.edu or clark.cl.li@gmail
@version:    1.0

@create:    April 18, 2018
@updated:   April 20, 2018

Algorithm: PerceptronTrain(D, MaxIter) 
1: wd ← 0, for all d = 1 . . . D        # initialize weights
2: b ← 0                                # initialize bias
3: for iter = 1 . . . MaxIter do
4:      for all (x,y) ∈ D do
5:        a ← ∑d=1~D wd xd + b            # compute activation for this example
6:        if ya ≤ 0 then
7:            wd ← wd + yxd, for all d = 1 ... D    # update weights
8:            b ← b + y                    # update bias
9:         end if
10:    end for
11: end for 
12: return w0, w1, ..., wD, b

Algorithm: PerceptronTest(w0, w1, ..., wD, b, ˆx) 
1: a ← ∑D d=1 wd xˆ_d + b             # compute activation for the test example
2: return sign(a)


Algorithm: AveragedPerceptronTrain(D, MaxIter) 
1: w ← <0, 0, . . . 0>, b ← 0         # initialize weights and bias
2: u ← <0, 0, . . . 0>, β ← 0         # initialize chased weights and bias
3: c ← 1                              # initialize example counter to one
4: for iter = 1 . . . MaxIter do
5:    for all (x,y) ∈ D do
6:        if y(w · x + b) ≤ 0 then
7:            w ← w + y x              # update weights
8:            b ← b + y                # update bias
9:            u ← u + y c x            # update cached weights
10:           β ← β + y c              # update cached bias
11:        end if
12:        c ← c + 1                   # increment counter regardless of update
13:    end for
14: end for 
15: return w - 1/c u, b - 1/c β    # return averaged weights and bias

'''
from __future__ import print_function 
from __future__ import division

__all__ = []
__version__ = 1.0
__date__ = '2018-04-18'
__updated__ = '2018-04-20'

import sys, os
import collections
import math, json, re
from datetime import datetime
import numpy as np
from random import seed, randrange

#Reference F1 score: 0.88 for vanilla perceptron and 0.89 for the averaged perceptron,

DEBUG = 0 # 1 = print debug information, 2=detail steps information 
PRINT_TIME = 0 # 0= disable, 1 = print time stamps, 2 = print detail time stamps
TOKEN_DELIMITER = ' ' #the splitter for each token ( word/tag ).
COLUMNS = 4 # How many columns of data store in the training set.
LOW_FREQ_OBSERVATION_THRESHOLD = 2 # words appear more than or equal to the number of times will be reserved
HIGH_FREQ_OBSERVATION_THRESHOLD =1000 # words appear less than or equal to the number of times will be reserved
FOLDS = 10 # folds = None or folds >= 1, folds=1=random sample data without validation set.
TEST_RATIO = 0.1 # Holdout ratio, TEST_RATIO = 0 = sequance sample data without validate set
ITERATION = 30
CONVERAGE = 0.0001
PATIENT = 5
SEED = 9999999
ASCII_ONLY = True
REMOVE_STOPWORDS = True
REMOVE_PUNCTUATION = True
# NLTK stop words
'''
STOP_WORDS = ['ourselves', 'he', 've', 'and', 'm', 'shan', 'having', 'an', 'other', 'wasn', 'me', 'had', 'why', 'up', 'same', 'these',\
              'be', 'did', 'some', 'few', 'she', 'between', 'for', 'as', 'weren', 'most', 'from', 'no', 'in', 'there', 'but', 'before',\
              'about', 'what', 'then', 'her', 'any', 'more', 'of', 'once', 'now', 'or', 'y', 'their', 'don', 'who', 'which', 'at', 'to',\
              'isn', 'each', 'own', 'because', 'myself', 'll', 't', 're', 'wouldn', 'were', 'doesn', 'until', 'such', 'both', 'only',\
              'we', 'with', 'ma', 'against', 'couldn', 'they', 'doing', 'needn', 'your', 'too', 'them', 'aren', 'yours', 'didn', 'that',\
              'is', 'mustn', 'should', 'being', 'i', 'on', 'if', 'mightn', 'when', 'down', 'haven', 'where', 'it', 'than', 'how',\
              'itself', 'our', 'so', 'himself', 'shouldn', 'above', 'you', 'ain', 'my', 'can', 'after', 'while', 'the', 'him', 'hasn',\
              'a', 'been', 's', 'will', 'ours', 'into', 'yourself', 'here', 'further', 'by', 'yourselves', 'his', 'whom', 'do', 'over',\
              'under', 'very', 'was', 'hadn', 'again', 'theirs', 'not', 'nor', 'those', 'this', 'below', 'does', 'all', 'has', 'during',\
              'am', 'hers', 'd', 'off', 'have', 'through', 'out', 'herself', 'just', 'its', 'o', 'themselves', 'won', 'are']
# Stanford NLP stop words
STOP_WORDS = ["'ll", "'s", "'m", 'a', 'about', 'above', 'after', 'again',\
              'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',\
              'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do' , 'does',\
              "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't",\
              'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself',\
              'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',\
              "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',\
              'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's",\
              'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves',\
              'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through',\
              'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't",\
              'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with',\
              "won't", 'wourld', 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',\
              'yourselves', '###', 'return', 'arent', 'cant', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent', 'hes',\
              'heres', 'hows', 'im', 'isnt', 'its', 'lets', 'mustnt', 'shant', 'shes', 'shouldnt', 'thats', 'theres', 'theyll',\
              'theyre', 'theyve', 'wasnt', 'were','werent', 'whats', 'whens', 'wheres', 'whos', 'whys', 'wont', 'wouldnt', 'youd',\
              'youll', 'youre', 'youve']
'''              
# Stanford NLP + NLTK stop words, remove but, add us
STOP_WORDS = ['ve', 'm', 'shan', 'wasn', 'weren', 'y','don', 'isn', 'll', 't', 're', 'wouldn', 'dosen', 'ma', 'couldn', 'needn', 'aren',\
              'didn','mustn', 'mightn', 'haven', 'shouldn', 'ain', 'hasn', 's', 'hadn', 'd', 'o', 'won', 'but',\
              "'ll", "'s", "'m", 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't",\
              'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'by', 'can', "can't", 'cannot',\
              'could', "couldn't", 'did', "didn't", 'do' , 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for',\
              'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',\
              "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is',\
              "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off',\
              'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd",\
              "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them',\
              'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through',\
              'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't",\
              'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with',\
              "won't", 'wourld', 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',\
              'yourselves', '###', 'return', 'arent', 'cant', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent', 'hes',\
#               'heres', 'hows', 'im', 'isnt', 'its', 'lets', 'mustnt', 'shant', 'shes', 'shouldnt', 'thats', 'theres', 'theyll',\
              'theyre', 'theyve', 'wasnt', 'were','werent', 'whats', 'whens', 'wheres', 'whos', 'whys', 'wont', 'wouldnt', 'youd',\
              'youll', 'youre', 'youve', 'us']

PUNCTUATION = ['!!','?!','??','!?','`','``',"''", ',', '.', ':', ';', '"', "'", '?', '<', '>', '{', '}', '[', ']', '+', '-', '(',\
              ')', '&', '%', '$', '@', '!', '^', '#', '*', '..', '...']
VANILLA_MODEL_FILE_NAME = './vanillamodel.txt'
AVERAGED_MODEL_FILE_NAME = './averagedmodel.txt'

def get_input(file_name):
    document = []

    try: 
        with open(file_name, 'r', encoding='utf-8') as _fp:
            for _each_line in _fp:
                _each_line =_each_line.strip()
                document.append(_each_line)                    
        return document 
    except IOError as _err:
        if (1): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()

def print_list(l):
    for i in l:
        print(i)
      

def get_feature_matrix(documents):
    '''
    Find total vocabulary we have from those reviews and create document vector for each review. 
        word_dict = {'word0': 0, 'word1': 1,....'wordn', n} where the 0 ... n is the index for word0 ... wordn.
        document_matrix = [
            [0, 0, 1, 5, ...1, 0]    # First document / review with word distribution. There are 1 word2, 5 word3, ... 1 wordn-1 
            [1, 0, 0, 1, ...0, 0]    # Second document / review with word distribution. There are 1 word0, 1 word3, ... 
            [2, 10, 1, 2, ...3, 0]
            ...
            [0, 11, 0, 0, ...9, 1]
        ]   
     '''
    classes_dict_list = [{}, {}]    # [{'Fake': -1, 'True': 1}, {'Neg': -1, 'Pos': 1}]
    label = []
    data = []    
    word_dict = collections.OrderedDict() # dictionary for each word
    document_matrix = [] # Store vector of documents 
    word_doc_count = {} #{'word1': number_of_documents_exist_word1, 'word2': ...} for IDF calculations
    init_binary_classification = -1
    
    review, sentences = '', ''
    tokenize = tokenizer(STOP_WORDS, PUNCTUATION)
             
    for _each_line in documents: 
        review = _each_line.rstrip('\n').split(TOKEN_DELIMITER, COLUMNS-1)
        
        # Get review from (columns-1)th column of text content file.
        sentences = tokenize.get_wordlist(review[COLUMNS-1], ascii_only=ASCII_ONLY, remove_stopwords=REMOVE_STOPWORDS, remove_punctuation = REMOVE_PUNCTUATION)
        if DEBUG > 1:
            print (sentences)
        
        label.append([review[1], review[2]])
        data.append(sentences)
        
        # Collect how many classes we have to identify and give it '1' or '-1' as label value.
        if (review[1] in classes_dict_list[0]) and (review[2] in classes_dict_list[1]):
            pass
        elif (review[1] not in classes_dict_list[0]) and (review[2] not in classes_dict_list[1]):
            classes_dict_list[0][review[1]] = init_binary_classification
            classes_dict_list[1][review[2]] = init_binary_classification
            init_binary_classification *= -1
        elif review[1] not in classes_dict_list[0]:
            classes_dict_list[0][review[1]] = init_binary_classification
        elif review[2] not in classes_dict_list[1]:
            classes_dict_list[1][review[2]] = init_binary_classification            

### TF-IDF feature
#     tfidf = tf_idf(LOW_FREQ_OBSERVATION_THRESHOLD, HIGH_FREQ_OBSERVATION_THRESHOLD)
#     word_dict, document_matrix, word_doc_count = tfidf.get_dictionary_n_document_matrix(data)
#     # Convert the label from text (Fake, True, Neg, Pos) to value (+1, -1)
#     label = np.array([ [classes_dict_list[0][cf[0]], classes_dict_list[1][cf[1]]]  for cf in label ])
#
                  
    wc = word_counts(LOW_FREQ_OBSERVATION_THRESHOLD, HIGH_FREQ_OBSERVATION_THRESHOLD)
    word_dict, document_matrix = wc.get_dictionary_n_document_matrix(data)
    # Convert the label from text (Fake, True, Neg, Pos) to value (+1, -1)
    label = np.array([ [classes_dict_list[0][cf[0]], classes_dict_list[1][cf[1]]]  for cf in label ])
  
    return word_dict, document_matrix, classes_dict_list, label, word_doc_count

class tf_idf(object):
    def __init__(self, LOW_FREQ_OBSERVATION_THRESHOLD = None, HIGH_FREQ_OBSERVATION_THRESHOLD = None):
        self.word_dict = collections.OrderedDict() # dictionary for each word {'word1': no of documents have word1, 'word2':...}
        self.document_matrix = [] # Store vector of documents # store tf then calculate tf-idf
        self.word_doc_count = {} #{'word1': number_of_documents_exist_word1, 'word2': ...} for IDF calculations
        self.low_freq_threshold = LOW_FREQ_OBSERVATION_THRESHOLD
        self.high_freq_threshold = HIGH_FREQ_OBSERVATION_THRESHOLD

    
    def get_dictionary_n_document_matrix(self, documents):
        word_dict = self.word_dict
        doc_matrix = self.document_matrix
        high_threshold = self.high_freq_threshold
        low_threshold = self.low_freq_threshold
        N = len(documents) # total number of documents
        word_count = {} #{'word1': count1, 'word2': count2,...}
        word_doc_count = {} #{'word1': number_of_documents_exist_word1, 'word2': ...} for IDF calculations

        # Build word counts dictionary
        for doc in documents:
            doc_vector = []
            word_set = set()
            for word in doc:
                word_count[word]  = word_count.get(word, 0) + 1
                word_set.add(word)
            for word in word_set:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
                     
        if DEBUG > 0 : print('Word_count=%s'%(word_count))
        for doc in documents:
            doc_vector = [0] * len(word_dict)
            for word in doc:
                # detect low or high frequency words 
                if word_count[word] >= low_threshold and word_count[word] <= high_threshold:
                    if word in word_dict: # if word in dictionary
                        idx = word_dict[word] # Get index
                    else: # Create word index into dictionary
                        idx = len(word_dict)
                        word_dict[word] = idx
                    if len(doc_vector) > idx:
                        doc_vector[idx] += 1 # add word counts
                    else: # A new word in the document 
                        doc_vector.insert(idx, 1) # add one word in the index position of the word.   
                else: # skip low or high frequency words 
                    pass
            # Calculate TF-IDF
            nw = sum(doc_vector)
            for word in doc:
                if word_count[word] >= low_threshold and word_count[word] <= high_threshold:
                    _i = word_dict[word]
                    doc_vector[_i] = doc_vector[_i]/nw * math.log(N/word_doc_count[word])            
            doc_matrix.append(doc_vector)    
        
        # initial a matrix with row no = no. of reviews and columns = no. of vocabulary
        _m = np.zeros([len(doc_matrix),len(word_dict)]) 
        for i,doc in enumerate(doc_matrix):
            _m[i][0:len(doc)] = doc
                
        self.word_dict = word_dict
        self.document_matrix = np.matrix(_m)
        self.word_doc_count = word_doc_count
        if DEBUG > 0 : print (self.document_matrix) 
        return self.word_dict, self.document_matrix, self.word_doc_count


class word_counts(object):
    def __init__(self, LOW_FREQ_OBSERVATION_THRESHOLD = None, HIGH_FREQ_OBSERVATION_THRESHOLD = None):
        self.word_dict = collections.OrderedDict() # dictionary for each word
        self.document_matrix = [] # Store vector of documents 
        self.low_freq_threshold = LOW_FREQ_OBSERVATION_THRESHOLD
        self.high_freq_threshold = HIGH_FREQ_OBSERVATION_THRESHOLD

    
    def get_dictionary_n_document_matrix(self, documents):
        word_dict = self.word_dict
        doc_matrix = self.document_matrix
        high_threshold = self.high_freq_threshold
        low_threshold = self.low_freq_threshold
        word_count = {} #{'word1': count1, 'word2': count2,...}

        # Build word counts dictionary
        for doc in documents:
            doc_vector = []
            for word in doc:
                word_count[word]  = word_count.get(word, 0) + 1         
        if DEBUG > 0 : print('Word_count=%s'%(word_count))
        for doc in documents:
            doc_vector = [0] * len(word_dict)
            for word in doc:
                # detect low or high frequency words 
                if word_count[word] >= low_threshold and word_count[word] <= high_threshold:
                    if word in word_dict: # if word in dictionary
                        idx = word_dict[word] # Get index
                    else: # Create word index into dictionary
                        idx = len(word_dict)
                        word_dict[word] = idx
                    if len(doc_vector) > idx:
                        doc_vector[idx] += 1 # add word counts
                    else: # A new word in the document 
                        doc_vector.insert(idx, 1) # add one word in the index position of the word.   
                else: # skip low or high frequency words 
                    pass
            doc_matrix.append(doc_vector)    
        
        # initial a matrix with row no = no. of reviews and columns = no. of vocabulary
        _m = np.zeros([len(doc_matrix),len(word_dict)]) 
        for i,doc in enumerate(doc_matrix):
            _m[i][0:len(doc)] = doc
                
        self.word_dict = word_dict
        self.document_matrix = np.matrix(_m)
        if DEBUG > 0 : print (self.document_matrix) 
        return self.word_dict, self.document_matrix

class Perceptron(object):
    '''
    Algorithm: PerceptronTrain(D, MaxIter) 
    1: wd ← 0, for all d = 1 . . . D        # initialize weights
    2: b ← 0                                # initialize bias
    3: for iter = 1 . . . MaxIter do
    4:      for all (x,y) ∈ D do
    5:        a ← ∑d=1~D wd xd + b            # compute activation for this example
    6:        if ya ≤ 0 then
    7:            wd ← wd + yxd, for all d = 1 ... D    # update weights
    8:            b ← b + y                    # update bias
    9:         end if
    10:    end for
    11: end for 
    12: return w0, w1, ..., wD, b
    
    Algorithm: PerceptronTest(w0, w1, ..., wD, b, ˆx) 
    1: a ← ∑D d=1 wd xˆ_d + b             # compute activation for the test example
    2: return sign(a)
    
    
    Algorithm: AveragedPerceptronTrain(D, MaxIter) 
    1: w ← <0, 0, . . . 0>, b ← 0         # initialize weights and bias
    2: u ← <0, 0, . . . 0>, β ← 0         # initialize chased weights and bias
    3: c ← 1                              # initialize example counter to one
    4: for iter = 1 . . . MaxIter do
    5:    for all (x,y) ∈ D do
    6:        if y(w · x + b) ≤ 0 then
    7:            w ← w + y x              # update weights
    8:            b ← b + y                # update bias
    9:            u ← u + y c x            # update cached weights
    10:           β ← β + y c              # update cached bias
    11:        end if
    12:        c ← c + 1                   # increment counter regardless of update
    13:    end for
    14: end for 
    15: return w - 1/c u, b - 1/c β    # return averaged weights and bias
    '''    

    def __init__ (self, algorithm=None, iter=None):
        self.iteration = iter
        self.set_algorithm(algorithm)
        self.weights = None
        self.bias = None
        
    def set_algorithm(self, algorithm):
        if algorithm == 'vanilla':
            self.execute = self.vanilla_perceptron
        elif algorithm == 'averaged':
            self.execute = self.averaged_perceptron
        else:
            pass

    def load_model(self, weights, bias):   
        self.weights = np.array(weights)  
        self.bias = bias
    
    def get_fold_index(self, len_data, folds):
        dataset_split = []
        sample_index = [i for i in range(len_data)]
        fold_size = int(len_data / folds)
        for i in range(folds):
            fold = []
            while len(fold) < fold_size:
                index = randrange(len(sample_index))
#                 index = 0 #If follow the original data sequence
                fold.append(sample_index.pop(index))
            dataset_split.append(fold)
        return dataset_split

    def get_kfolds_training_validate_sets(self, iter, folds_sample):
        '''
        For fold = 3, folds_sample = [[7,1,3], [5,8,0], [2,4,6]]
        '''
        folds = len(folds_sample)
        training_index = []
        validate_index = []
        if folds == 1:
            training_index = folds_sample[folds-1]
            if DEBUG > 1: print(training_index)
        else:
            for i in range(folds):
                if i != iter%folds:
                    training_index.extend(folds_sample[i])
                else:
                    validate_index.extend(folds_sample[i])
            if DEBUG > 1: print('iter=%d, training=%s'%(iter, training_index))
            if DEBUG > 1: print('iter=%d, validate=%s'%(iter, validate_index))
        return training_index, validate_index

                  
    def vanilla_perceptron(self, training, training_labels, validate=None, validate_labels=None, folds=None, patient=0):
        '''
        This perceptron classifiers learner algorithm of vanilla perceptron.
        Algorithm: PerceptronTrain(D, MaxIter) 
        1: wd ← 0, for all d = 1 . . . D        # initialize weights
        2: b ← 0                                # initialize bias
        3: for iter = 1 . . . MaxIter do
        4:      for all (x,y) ∈ D do
        5:        a ← ∑d=1~D wd xd + b            # compute activation for this example
        6:        if ya ≤ 0 then
        7:            wd ← wd + yxd, for all d = 1 ... D    # update weights
        8:            b ← b + y                    # update bias
        9:         end if
        10:    end for
        11: end for 
        12: return w0, w1, ..., wD, b
        
        Algorithm: PerceptronTest(w0, w1, ..., wD, b, ˆx) 
        1: a ← ∑D d=1 wd xˆ_d + b             # compute activation for the test example
        2: return sign(a)
        '''             
        X = training
        Y = training_labels
       
        max_f1 = 0
        _no_improve = 0
        max_weights = None
        max_bias = None
        
        weights = np.zeros(shape=(1, X.shape[1])) # Get the number of vocabularies
        bias = 0
        training_index, validate_index = 0, 0

        if folds is not None and folds > 1:  #K-folds approach
            folds_sample = self.get_fold_index(len(X), folds)  
            self.iteration = folds          
#         print (folds_sample)
        for _it in range(self.iteration):   
            if DEBUG > 0: print('iteration=%d'%(_it))
            if folds is not None:  #K-folds approach
                if folds == 1: # random select data from training set
                    folds_sample = self.get_fold_index(len(X), folds)                    
                training_index, validate_index = self.get_kfolds_training_validate_sets( _it, folds_sample)
            else: # general training / validate set approach
                training_index, validate_index = [i for i in range(len(training))], [i for i in range(len(validate))]
                
            for i in training_index:
                x = X[i]
                y = Y[i]
                _a = np.dot(weights, x.transpose()) + bias
                if y*_a <= 0:
                    weights = weights + np.dot(y, x)
                    bias = bias + y
                if DEBUG > 1 : print('iteration:%d, y=%f, _a=%f, x=%s, b=%f, w=%s'%(_it, y, _a, str(x), bias, str(weights)))

            self.weights = weights
            self.bias = bias       
                           
            if (folds is not None and folds > 1):
                vX = [X[i] for i in validate_index]
                vY = [Y[i] for i in validate_index]              
            elif validate is not None and len(validate) > 0 and validate_labels is not None and len(validate_labels) > 0:
                vX = validate
                vY = validate_labels   
            else:
                vX = X
                vY = Y
                 
            max_f1, _avg_f1, max_weights, max_bias, _no_improve = self.cross_validate(vX, vY, max_f1, max_weights, max_bias, _no_improve)
            if _no_improve >= patient and (folds is None or _it > folds): # Make sure k-folds are passed
                if DEBUG > 0: print('Escape: _final_avg_f1=%f, max_avg_f1=%f\n'%(_avg_f1, max_f1)) 
                break             
        self.weights = max_weights
        self.bias = max_bias
#         return [self.weights.tolist(), self.bias.tolist()]        
        return self.weights, self.bias
    
    def averaged_perceptron(self, training, training_labels, validate=None, validate_labels=None, folds=1, patient=0):
        '''
        This perceptron classifiers learner algorithm of averaged perceptron.
    
        Algorithm: AveragedPerceptronTrain(D, MaxIter) 
        1: w ← <0, 0, . . . 0>, b ← 0         # initialize weights and bias
        2: u ← <0, 0, . . . 0>, β ← 0         # initialize chased weights and bias
        3: c ← 1                              # initialize example counter to one
        4: for iter = 1 . . . MaxIter do
        5:    for all (x,y) ∈ D do
        6:        if y(w · x + b) ≤ 0 then
        7:            w ← w + y x              # update weights
        8:            b ← b + y                # update bias
        9:            u ← u + y c x            # update cached weights
        10:           β ← β + y c              # update cached bias
        11:        end if
        12:        c ← c + 1                   # increment counter regardless of update
        13:    end for
        14: end for 
        15: return w - ((1/c)*u), b - ((1/c)* β)    # return averaged weights and bias
            
        '''
        X = training
        Y = training_labels      
        max_f1 = 0
        _no_improve = 0    
        max_weights = None
        max_bias = None
        
        weights = np.zeros(shape=(1, X.shape[1])) # Get the number of vocabularies
        bias = 0
        training_index, validate_index = 0, 0        
        
        u = np.zeros(shape=(1, X.shape[1])) # Get the number of vocabularies
        b = 0
        c = 1
        
        if folds is not None and folds >1:  #K-folds approach
            folds_sample = self.get_fold_index(len(X), folds)    
            self.iteration = folds 
        for _it in range(self.iteration):
            if DEBUG > 0: print('iteration=%d'%(_it))
            if folds is not None:  #K-folds approach
                if folds == 1: # random select data from training set
                    folds_sample = self.get_fold_index(len(X), folds)                  
                training_index, validate_index = self.get_kfolds_training_validate_sets( _it, folds_sample)
            else: # general training / validate set approach
                training_index, validate_index = [i for i in range(len(training))], [i for i in range(len(validate))]
            
            for i in training_index:
                x = X[i]
                y = Y[i]
                _a = np.dot(weights, x.transpose()) + bias
                if y*_a <= 0:
                    weights = weights + np.dot(y, x)
                    bias = bias + y
                    
                    u = u + np.dot(y*c, x)
                    b = b + y*c
                c += 1
                if DEBUG > 1 : ('iteration:%d, y=%f, _a=%f, x=%s, b=%f, w=%s'%(_it, y, _a, str(x), bias, str(weights)))
            self.weights = weights-(1/c)*u
            self.bias = bias-(1/c)*b
            
            if (folds is not None and folds > 1):
                vX = [X[i] for i in validate_index]
                vY = [Y[i] for i in validate_index]              
            elif validate is not None and len(validate) > 0 and validate_labels is not None and len(validate_labels) > 0:
                vX = validate
                vY = validate_labels   
            else:
                vX = X
                vY = Y
                
            max_f1, _avg_f1, max_weights, max_bias, _no_improve = self.cross_validate(vX, vY, max_f1, max_weights, max_bias, _no_improve)
            if _no_improve >= patient and (folds is None or _it > folds): # Make sure k-folds are passed
                if DEBUG > 0: print('Escape: _final_avg_f1=%f, max_avg_f1=%f\n'%(_avg_f1, max_f1)) 
                break
        self.weights = max_weights
        self.bias = max_bias
#         return [self.weights.tolist(), self.bias.tolist()]
        return self.weights, self.bias
    
    def cross_validate(self, vX, vY, max_f1, max_weights, max_bias, no_improve):
            pY = self.predict(vX)
            _avg_f1, _ = classification_report(vY, pY, print_results=False)    
            if (_avg_f1 - max_f1) >= CONVERAGE:
                max_f1 = _avg_f1      
                max_weights = self.weights
                max_bias = self.bias
                                
                no_improve = 0  
                if DEBUG > 0: print(' _avg_f1=%f'%(_avg_f1))
            elif (_avg_f1 - max_f1) < CONVERAGE:
                no_improve += 1
                if DEBUG > 0: print('May skip _avg_f1=%f, no_improve=%d'%(_avg_f1, no_improve))
#             elif (_avg_f1 - max_f1) == 0 : # No improve in avg f1, but the weights and bias may more reliable      
#                 max_weights = self.weights
#                 max_bias = self.bias                
#                 no_improve += 1              
#                 if DEBUG > 0: print('May skip  _avg_f1=%f, no_improve=%d'%( _avg_f1, no_improve))   
            else:
                pass
            return max_f1, _avg_f1,  max_weights, max_bias, no_improve

    def predict(self, data, class_dict=None):
        '''
        data = multiple test cases
        class_dict to convert digital results to text classifications
        
        Algorithm: PerceptronTest(w0, w1, ..., wD, b, ˆx) 
        1: a ← ∑D d=1 wd xˆ_d + b             # compute activation for the test example
        2: return sign(a)
        
        '''
        weights = self.weights
        bias = self.bias
        Y = []
        X = data
        
        for x in X:
            _a = np.dot(weights, x.transpose()) + bias
            if _a <=0 :
                y = -1
            else:
                y = 1
            if class_dict != None:
                y = class_dict[y]
            else:
                pass
            Y.append(y)
        return Y
               
class tokenizer(object):

    def __init__(self, stopword = STOP_WORDS, punctuation = PUNCTUATION):
        self.stop_words = stopword
        self.punctuation = punctuation
        
    def get_wordlist(self, sentence, ascii_only=True, remove_stopwords=False, remove_punctuation=False ):

        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.

        # Remove non-letters, we may remark this line and see different filtering approach. ####
        if ascii_only:
            sentence = re.sub("[^a-zA-Z]"," ", sentence)
        else:
            pass
        # Convert all characters to lower case and split them
        words = sentence.lower().split()

        # Optionally remove stop words (false by default)
        if remove_stopwords and remove_punctuation:
            wordlist = [w for w in words if (not w in self.stop_words and not w in self.punctuation)]
        elif remove_stopwords:
            wordlist = [w for w in words if (not w in self.stop_words)]
        elif remove_punctuation:
            wordlist = [w for w in words if (not w in self.punctuation)]
        else:
            wordlist = words

        # Return a word list
        return wordlist

    # Define a function to split a review into parsed sentences
    def document_to_sentences(self, document, ascii_only=True, remove_stopwords=False, remove_puncutation=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        raw_sentences = document.rstrip('\n')

        #
        # Loop over each sentence
        sentences = []
        for _sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(self.get_wordlist(_sentence, ascii_only, remove_stopwords, remove_puncutation ))
            else:
                pass
        # Return the list of sentences which are lists of words,
        return sentences


def classification_report(truth_list, predict_list, print_results=False):
    '''
    results = {class1:{TP:count, FP: count, FN:count}, ...}
    '''
    count_results = {}
    score_results = {}
    avg_f1 = 0
    for i, predict in enumerate(predict_list):
        if predict == truth_list[i]:
            _tmp_dict = count_results.get(predict, {'TP': 0})
            _tmp_dict['TP']=_tmp_dict.get('TP', 0) + 1
            count_results[predict] = _tmp_dict
        elif predict != truth_list[i]:
            _tmp_dict = count_results.get(predict, {'FP': 0})
            _tmp_dict['FP']=_tmp_dict.get('FP', 0) + 1
            count_results[predict] = _tmp_dict
 
            _tmp_dict = count_results.get(truth_list[i], {'FN': 0})
            _tmp_dict['FN']=_tmp_dict.get('FN', 0) + 1
            count_results[truth_list[i]] = _tmp_dict            
                     
    for key, result in count_results.items():
        precision = result.get('TP', 0)/(result.get('TP', 0)+result.get('FP', 1)) 
        support = result.get('TP', 0)+result.get('FN', 0)
        recall = result.get('TP', 0)/support
        if (precision+recall) == 0:
            F1 = 0
        else:
            F1 = 2*precision*recall/(precision+recall)
        score_results[key] = {'precision': precision, 'recall': recall, 'f1-score': F1, 'support': support}
        if print_results == True: print ('class:%s, precision=%f, recall=%f, f1-score=%f, support=%d'%( key, precision, recall, F1, support ))
    total_f1 = 0
    total_support = 0
    for key, result in score_results.items():
        total_f1 += result['f1-score']*result['support']
        total_support += result['support']
    if total_support != 0:
        avg_f1 = total_f1/total_support
    else:
        avg_f1 = None
    if print_results == True: print ('average f1=%f'%(avg_f1))    
    return avg_f1, score_results  

def train_test_split(X, Y, test_rate=0.1):
    training_size = int(len(X)*(1-test_rate))
  
    return  X[0:training_size, :], X[training_size:, :], Y[0:training_size], Y[training_size:]

def save_model(model_file_name, model):
    #Save the model to model_file_name
    with open(model_file_name, 'w', encoding='utf-8') as fp:
        json.dump(model, fp , indent=1, ensure_ascii=False)
                
'''
    Main program for the prior, and evident probabilities tables generate for Naive Bayes class execution.

'''
   
def main(input_doc):
    parameters_list = [] #[class1[weights, bias], class2[weights, bias]]
    test_ratio = TEST_RATIO
    
    if PRINT_TIME : print ('perceplearn.get_input=>Start=>%s'%(str(datetime.now())))
    document = get_input(input_doc)
    
    ##################

    if DEBUG > 1: print_list(document)
    if PRINT_TIME : print ('perceplearn.get_probabilities_tables=>Start=>%s'%(str(datetime.now())))    
    word_dict, document_matrix, classes_list, label, word_doc_count = get_feature_matrix(document)


    # Predict by vanilla model.
    p = Perceptron('vanilla', ITERATION)
    parameters_list = [] #[class1[weights, bias], class2[weights, bias]]
    for i, class_label in enumerate(classes_list):
        if DEBUG > 0: print ('vanilla: class_label=%s'%(class_label))
        # Prepare data
        training_index, validate_index, training_label, validate_label = train_test_split( document_matrix, np.array(label)[:, i], test_rate=test_ratio)  
        weights1, bias1 = p.execute(training_index, training_label, validate_index, validate_label, folds=None, patient=PATIENT)
        weights2, bias2 = p.execute(document_matrix, np.array(label)[:, i], None, None, folds=1, patient=PATIENT)
        weights3, bias3 = p.execute(document_matrix, np.array(label)[:, i], None, None, folds=FOLDS, patient=PATIENT)        
        weights = (weights1+weights2+weights3)/3
        bias = (bias1+bias2+bias3)/3
        parameters_list.append( [weights.tolist(), bias.tolist()] )
    
    save_model(VANILLA_MODEL_FILE_NAME, [word_dict, parameters_list, classes_list, word_doc_count])
    
    # Predict by averaged model.
    p.set_algorithm('averaged')
    parameters_list = []
    for i, class_label in enumerate(classes_list):
        if DEBUG > 0: print ('averaged: class_label=%s'%(class_label))
        training_index, validate_index, training_label, validate_label = train_test_split( document_matrix, np.array(label)[:, i], test_rate=test_ratio)  
#         parameters_list.append(p.execute(training_index, training_label, validate_index, validate_label, folds=FOLDS, patient=PATIENT))  
        weights1, bias1 = p.execute(training_index, training_label, validate_index, validate_label, folds=None, patient=PATIENT)
        weights2, bias2 = p.execute(document_matrix, np.array(label)[:, i], None, None, folds=1, patient=PATIENT)
        weights3, bias3 = p.execute(document_matrix, np.array(label)[:, i], None, None, folds=FOLDS, patient=PATIENT)        
        weights = (weights1+weights2+weights3)/3
        bias = (bias1+bias2+bias3)/3
        parameters_list.append( [weights.tolist(), bias.tolist()] )    
    save_model(AVERAGED_MODEL_FILE_NAME, [word_dict, parameters_list, classes_list, word_doc_count])
        
    if PRINT_TIME : print ('perceplearn.get_probabilities_tables=>End=>%s'%(str(datetime.now())))   
                   
if __name__ == '__main__':
    '''
    Main program.
        1. Read the training file from train-labeled.txt as default.
        2. Using each vocabulary as features   
        3. Construct dictionary and word counting for each review.
        4. Perform Percepton algorithm to calculate weights and b.
            iterations = 30
            early stop criteria: 

        5. To prevent under flow, we will use log probability to calculus.
             log P(H_j|E) ~ log(P(H_j)) + SUM_i(log P(E_i|H_j)) - SUM_i(log P(E_i))
        6. Construct evidence probability table for each word. 
        7. Store these probability tables in nboutput.txt for nbclassify3.py to read and perform classification tasks.
    '''      
    
    # Get input and output parameters
    if len(sys.argv) != 2:
        print('Usage: ' + sys.argv[0] + ' /path/to/inputfile ')
        sys.exit(1)
    
    seed(SEED)
    # Assign the input and output variables
    INPUT_FILE = sys.argv[1]
    main (INPUT_FILE)
