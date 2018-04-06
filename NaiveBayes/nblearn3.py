#!/usr/bin/env python
# encoding: utf-8
'''
Natural Language Processing
Algorithm Name: Naive Bayes model implementation on sentiment reviews 
    to identify hotel reviews as either true or fake, and either positive or negative.

This is a program to implement of Naive Bayes model sentiment reviews for English.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2018 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    chenglil@usc.edu or clark.cl.li@gmail.com
@version:    1.2

@create:    March 27, 2018
@updated:   April 04, 2018

Tasks:
This task will write a naive Bayes classifier to identify hotel reviews as either true (True) or fake(Fake), and either positive(Pos) or negative(Neg). 
The program (nblearn3.py) will perform document split to the word level token as features to train for classification learning and generate probabilities tables into file.  
The argument is a single file containing the training data; the program will learn a naive Bayes model, 
and write the model parameters to a file called nbmodel.txt.

The classifier program (nbclassify3.py) will read the model file and reviews to perform classifications. 

Smoothing:
The solution uses add-one smoothing on the training data, and simply ignores unknown tokens in the test data.

Tokenization: 
The program splits each word as a basic token unit. Certain punctuation will be removed, and lower casing all the letters. 
Stop word will be ignored. (high-frequency or low-frequency tokens). 

Data:
A set of training and development data will be made available as separate files. 
1. The file (train-labeled.txt) is training data which include classification and reviews.
2. The file (dev-text.txt) is development set of data, with sequence no. and reviews which is separated by a new line. 
3. The file (dev-key.txt) is classifications for development set of data. 

Data Structures and Global variables:
1. MODEL_FILE_NAME = './nbmodel.txt'
2. TOKEN_DELIMITER = ' ' #the splitter for each token ( word ).
3. STOP_WORDS = ''
4. CLASSIFICATIONS = [['True', 'Fake'], ['Pos', 'Neg']] 
5. Prior Probability = {'True': review counts, 'Fake': review counts, 'Pos': review counts, 'Neg': review counts}
6. Posterior Probability = {'word_1': {'True': counts, 'Fake': counts, 'Pos': counts, 'Neg': counts}, {}...{}} 

Reference Performance:
 Pos F1=0.93, Neg F1=0.92, True F1=0.89, Fake F1=0.89, mean F1 = 0.9078.
'''
from __future__ import print_function 
from __future__ import division

import sys
import collections
import json, re
from datetime import datetime

DEBUG = 0 # 1 = print debug information, 2=detail steps information 
PRINT_TIME = 0 # 0= disable, 1 = print time stamps, 2 = print detail time stamps
MODEL_FILE_NAME = './nbmodel.txt'
TOKEN_DELIMITER = ' ' #the splitter for each token ( word/tag ).
COLUMNS = 4 # How many columns store in the training set.
LOW_FREQ_OBSERVATION_THRESHOLD = 2 # words appear more than or equal to the number of times in any one class will be reserved
HIGH_FREQ_OBSERVATION_THRESHOLD = 1000 # words appear less than or equal to the number of times in any one class will be reserved
LAMBDA = 0.1
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
              'heres', 'hows', 'im', 'isnt', 'its', 'lets', 'mustnt', 'shant', 'shes', 'shouldnt', 'thats', 'theres', 'theyll',\
              'theyre', 'theyve', 'wasnt', 'were','werent', 'whats', 'whens', 'wheres', 'whos', 'whys', 'wont', 'wouldnt', 'youd',\
              'youll', 'youre', 'youve', 'us']

PUNCTUATION = ['!!','?!','??','!?','`','``',"''", ',', '.', ':', ';', '"', "'", '?', '<', '>', '{', '}', '[', ']', '+', '-', '(',\
              ')', '&', '%', '$', '@', '!', '^', '#', '*', '..', '...']
OUTPUT_FILE_NAME = './nboutput.txt'
HIGH_FREQ_OBSERVATION_THRESHOLD

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
      

def get_probabilities_tables(documents):
    '''
    Find total classes we have to classify. This implementation support multi-categories classification
        classes_list = [catetory1[class1, class2], category2[class3, class4]]
    Calculate the probability tables of prior , and posterior probabilities.
    Store classifications count by distributions/documents into variable "classes_doc_count" in a dictionary data structure.
        Data structure: classes_doc_count=OrderedDict([('True', count1), ('Fake', count2), ('Pos', count3), ('Neg', count4)])
    Store evidence count by each class into variable "evidence_class_count" in a dictionary data structure.
        Data structure: evidence_class_count=OrderedDict([('Word1', {'True': count1, 'Fake': count2, 'Pos': count3, 'Neg': count4}),('Word2, {...}),...])
   
     '''
    classes_list = [set(), set()]
    label = []
    data = []
    
    prior_prob = collections.OrderedDict() # classification and prior probabilities dictionary
    posterior_prob = collections.OrderedDict() # Store evidence probability
    
    review, sentences = '', ''
    tokenize = tokenizer(STOP_WORDS, PUNCTUATION)
             
    _tmp_state_obs_dict = {} 
    for _each_line in documents: #row is x
        review = _each_line.rstrip('\n').split(TOKEN_DELIMITER, COLUMNS-1)

        sentences = tokenize.get_wordlist(review[COLUMNS-1], ascii_only=ASCII_ONLY, remove_stopwords=REMOVE_STOPWORDS, remove_punctuation = REMOVE_PUNCTUATION)
        if DEBUG > 0:
            print (sentences)

        # Prior probability counts based on documents
        label.append([review[1], review[2]])
        data.append(sentences)

        classes_list[0].add(review[1])
        classes_list[1].add(review[2])
        
    nb = Naive_Bayes()
    prior_prob, posterior_prob = nb.get_probability(data, label)
             

    # Convert set to list for JSON file export.        
    classes_list[0]=list(classes_list[0])
    classes_list[1]=list(classes_list[1])
    
    #Save the model to MODEL_FILE_NAME
    with open(MODEL_FILE_NAME, 'w', encoding='utf-8') as fp:
        json.dump([prior_prob, posterior_prob, classes_list], fp , indent=1, ensure_ascii=False)

    return prior_prob, posterior_prob, classes_list

class Naive_Bayes(object):
    '''
    Naive_Bayes model apply to classify reviews / documents.

        1. According to Bayes theorem, P(H|E) = P(E|H) * P(H) / P(E)
            where: 
            a. P(H) is prior probability of hypothesis H being true. In our case is the percentage rate of true review.
             (and it can also be false, negative, positive reviews)
            b. P(E) is the probability of evidence (regardless of the hypothesis). In our case is the review words.
            c. P(E|H) is the probability of the evidence given that hypothesis is true.
            d. P(H|E) is the probability of the hypothesis given that the evidence (review) is the there. It is posterior probability.   
        2. Construct P(H) probability table by reviews and labels.
        3. According to Naive Bayes classifier, assume each word in post is independent.
          P(H_j|Multiple Evidences) ~=  P(E_1| H_j)* P(E_2|H_j) ……*P(E_n|H_j) * P(H_j) / P(Multiple Evidences)
           where:
           a. H_j is one of our hypothesis (True, False, Positive, Negative) 
           b. E_1, E_2, ...E_n are words in our training set.
           c. P(Multiple Evidences) = P(E) = SUM_j(P(E|H_j)*P(H_j))
        4. To prevent under flow, we will use log probability to calculus.
             log P(H_j|E) ~ log(P(H_j)) + SUM_i(log P(E_i|H_j)) - SUM_i(log P(E_i))
             where SUM_i(log P(E_i)) is a probability of given sentence, so it is a constant.
             => log P(H_j|E) ∝ log(P(H_j)) + SUM_i(log P(E_i|H_j))
    '''

    def __init__(self, priorProbTable=None, posteriorProbTable=None, \
                 low_freq_threshold=LOW_FREQ_OBSERVATION_THRESHOLD, high_freq_threshold=HIGH_FREQ_OBSERVATION_THRESHOLD, \
                 lambda_value = LAMBDA):
        self.set_probability(priorProbTable, posteriorProbTable)
        self.low_freq_threshold = low_freq_threshold
        self.high_freq_threshold = high_freq_threshold
        self.lambda_value = lambda_value

    def set_probability(self, priorProbTable, posteriorProbTable):
        self.priorProbTable = priorProbTable
        self.class_numbers = len(priorProbTable) if priorProbTable is not None else 0
        self.posteriorProbTable = posteriorProbTable    
        
    def get_probability(self, data, classifications):
        each_class_evidence_counts = collections.OrderedDict() # word / evidence counts of each class    
        evidence_class_counts = collections.OrderedDict() # class counts of each word
        classes_line_counts = collections.OrderedDict() # document / data line / distribution counts of each class
        total_records = 0
        prior_prob = collections.OrderedDict() # classification and prior probabilities dictionary
        posterior_prob = collections.OrderedDict() # Store evidence probability

        for _i, category_list in enumerate(classifications):
            for _each_class in category_list:
                classes_line_counts[_each_class] = classes_line_counts.get(_each_class, 0) + 1 
                for _value in data[_i]: # check every word/token/value
                    # Evidence probability counts
                    # For True / Fake and Pos / Neg
                    each_class_evidence_counts[_each_class]  = each_class_evidence_counts.get(_each_class, 0) + 1
                    _tmp_dict = evidence_class_counts.get(_value, {_each_class: 0})
                    _tmp_dict[_each_class] = _tmp_dict.get(_each_class, 0) + 1
                    evidence_class_counts[_value] = _tmp_dict                            
                total_records += 1        
         
        each_class_evidence_counts, evidence_class_count = self.set_smoothing_and_remove_data(each_class_evidence_counts, evidence_class_counts, self.lambda_value, self.low_freq_threshold, self.high_freq_threshold)

        if DEBUG > 0:
            print('classes_line_counts=%s'%(classes_line_counts))
            print('each_class_evidence_counts=%s'%(each_class_evidence_counts))        
            print('evidence_class_counts=%s'%(evidence_class_counts))
            print('prior_prob=%s'%(prior_prob))
            print('posterior_prob=%s'%(posterior_prob)) 
                
        # Calculate prior probability
        for each_class, counts in classes_line_counts.items():
            prior_prob[each_class] = counts/total_records
        
        for each_word, classes_dict in evidence_class_count.items():
            _tmp_dict = {}
            for each_class, counts in classes_dict.items():
                _tmp_dict[each_class] = counts/each_class_evidence_counts.get(each_class)
            posterior_prob[each_word] = _tmp_dict
        
        return prior_prob, posterior_prob

    def set_smoothing_and_remove_data(self, each_class_evidence_count, evidence_class_count, lambda_value, low_freq_threshold, high_freq_threshold):
        '''
        Additional data for Good-Turing smoothing and feature engineering.
        scan each word and tag to accumulate tag, word counting information.
        '''
        tmp_evidence_class_count = collections.OrderedDict() # class counts of each word
        remove_word_flag = False
    
        for word, word_dict in evidence_class_count.items(): # check counter for each word 
            remove_word_flag = True
            for _each_class, counts in word_dict.items():
                if remove_word_flag == True and counts >= low_freq_threshold and counts <= high_freq_threshold:
                    remove_word_flag = False
                else:
                    pass
            if remove_word_flag == True:
                pass
            else:
                each_class_evidence_count, word_dict = self.laplace_smoothing(each_class_evidence_count, word_dict, lambda_value)
                tmp_evidence_class_count[word] = word_dict
        
            
        
        return each_class_evidence_count, tmp_evidence_class_count

    def laplace_smoothing(self, each_class_evidence_count, word_dict, lambda_value):
        '''
            apply laplace smoothing, default lunda = 1 = add one smoothing
            on the posterior probability table
            lunda should <= 1
        '''
        
        for each_class, counts in each_class_evidence_count.items():
            each_class_evidence_count[each_class] = counts + lambda_value
            word_dict[each_class] = word_dict.get(each_class, 0) + lambda_value            
        return each_class_evidence_count, word_dict
    def classify(self, data, classes_list):            
        '''
        This is an implementation of naive bayes algorithm for classification based on log probability.
        Assigning each sentence: P(s|class)=Π P(word|class), where s is sentence = word1, word2, word3...
        P(class|s) = P(class) * P(s|class) / P(s), where (s) is a given sentence, so the P(s) is a constant.
        Then P(class|s) is proportional to P(class) * P(s|class)
        => P(class|s) ∝ P(class) * P(s|class)
        log p(class|s) ∝ log p(class) + Sum log p(word|class)
        '''
        classifications = ''
        tmp_class = ''

        time_steps = len(data)
        if (time_steps == 0 ): #If input sentences are zero length.
            return ''
        else:
            pass
        if PRINT_TIME > 1 : print ('nbclassify.Calculation=>start=>%s'%(str(datetime.now())))
                          
        #Calculation
        for classes in classes_list: # For each category of classification
            best_score = None
            tmp_class = ''
            for each_class in classes:
                score = 0
                score = math.log(self.priorProbTable.get(each_class))
                for word in data: # word in sentence
                    _word_class = self.posteriorProbTable.get(word, None)
                    if _word_class is None: # Ignore unseen word
                        if DEBUG > 0 : print ('? unseen word=%s'%(word))  
                        pass
                    else:
                        score += math.log(_word_class.get(each_class))
                if DEBUG > 0 : print ('score=%f, each_class=%s'%(score, each_class))  
                if best_score is None:
                    best_score = score
                    tmp_class = each_class
                elif score > best_score :
                    best_score = score
                    tmp_class = each_class
            if DEBUG > 0 : print ('* best_score=%f, class=%s'%(best_score, tmp_class))  
                    
            classifications += tmp_class + ' '        
        
        return classifications.rstrip(' ')
    
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


        
'''
    Main program for the prior, and evident probabilities tables generate for Naive Bayes class execution.

'''
           
if __name__ == '__main__':
    '''
    Main program.
        1. Read the training file from train-labeled.txt as default.
        2. According to Bayes theorem, P(H|E) = P(E|H) * P(H) / P(E)
            where: 
            a. P(H) is prior probability of hypothesis H being true. In our case is the percentage rate of true review.
             (and it can also be false, negative, positive reviews)
            b. P(E) is the probability of evidence (regardless of the hypothesis). In our case is the review words.
            c. P(E|H) is the probability of the evidence given that hypothesis is true.
            d. P(H|E) is the probability of the hypothesis given that the evidence (review) is the there. It is posterior probability.   
        3. Construct P(H) probability table by reviews and labels.
        4. According to Naive Bayes classifier, assume each word in post is independent.
          P(H_j|Multiple Evidences) ~=  P(E_1| H_j)* P(E_2|H_j) ……*P(E_n|H_j) * P(H_j) / P(Multiple Evidences)
           where:
           a. H_j is one of our hypothesis (True, False, Positive, Negative) 
           b. E_1, E_2, ...E_n are words in our training set.
           c. P(Multiple Evidences) = P(E) = SUM_j(P(E|H_j)*P(H_j))
        5. To prevent under flow, we will use log probability to calculus.
             log P(H_j|E) ~ log(P(H_j)) + SUM_i(log P(E_i|H_j)) - SUM_i(log P(E_i))
        6. Construct evidence probability table for each word. 
        7. Store these probability tables in nboutput.txt for nbclassify3.py to read and perform classification tasks.
    '''      
    # Get input and output parameters
    if len(sys.argv) != 2:
        print('Usage: ' + sys.argv[0] + ' /path/to/inputfile ')
        sys.exit(1)
    
    # Assign the input and output variables
    INPUT_FILE = sys.argv[1]
    if PRINT_TIME : print ('nblearn.get_input=>Start=>%s'%(str(datetime.now())))
    document = get_input(INPUT_FILE)
    
    ##################

    if DEBUG > 0: print_list(document)
    if PRINT_TIME : print ('nblearn.get_probabilities_tables=>Start=>%s'%(str(datetime.now())))    
    priorProbTable, evidentProbTable, classes_list = get_probabilities_tables(document)
    if PRINT_TIME : print ('nblearn.get_probabilities_tables=>End=>%s'%(str(datetime.now())))    
    
