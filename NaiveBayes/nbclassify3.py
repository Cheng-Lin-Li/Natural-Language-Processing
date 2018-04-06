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
This task will write a naive Bayes classifier to identify hotel reviews as either true or fake, and either positive or negative. 
The program (nbclassify3.py) will perform document classification tasks. 
The classifier program (nbclassify3.py) will read the model file and apply to each review to perform classifications. 
All unseen words will be dropped in this program.

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
Data Structures and Global variables:
1. MODEL_FILE_NAME = './nbmodel.txt'
2. TOKEN_DELIMITER = ' ' #the splitter for each token ( word ).
3. STOP_WORDS = ''
4. CLASSIFICATION_CATEGORYS = [['True', 'Fake'], ['Pos', 'Neg']] 
'''


from __future__ import print_function 
from __future__ import division

import sys
import collections
import math, json, re
from datetime import datetime


DEBUG = 0 # 1 = print debug information, 2=detail steps information 
PRINT_TIME = 0 # 0= disable, 1 = print time stamps, 2 = print detail time stamps
MODEL_FILE_NAME = './nbmodel.txt'
TOKEN_DELIMITER = ' ' #the splitter for each token ( word/tag ).
TEST_COLUMNS = 2
LOW_FREQ_OBSERVATION_THRESHOLD = 2 # words appear more than or equal to the number of times in any one class will be reserved
HIGH_FREQ_OBSERVATION_THRESHOLD = 1000 # words appear less than or equal to the number of times in any one class will be reserved
LAMBDA = 0.1
ASCII_ONLY = True
REMOVE_STOPWORDS = True
REMOVE_PUNCTUATION = True

# Stanford NLP + NLTK stop words
STOP_WORDS = []
PUNCTUATION = []
OUTPUT_FILE_NAME = './nboutput.txt'

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


def set_output(outfile_name, output_content):
    i = 0
    try:
        l = len(output_content)
        with open(outfile_name, 'w', encoding='utf-8') as fp:
            for line in output_content:
                fp.write(line)
                if i < l-1:
                    fp.write('\n')
                i += 1
        fp.close()
    except IOError as _err:
        if (1): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()        
        
def load_parameters(file_name):
    # Load priorProbTable, posteriorProbTable probabilities tables generate for HMM class execution.
    priorProbTable = collections.OrderedDict()
    posteriorProbTable = collections.OrderedDict()
    
    try: 
        #Load the model from MODEL_FILE_NAME
        with open(file_name, 'r', encoding='utf-8') as fp:
            _prob_list = json.load(fp)
            priorProbTable = _prob_list[0]
            posteriorProbTable = _prob_list[1]
            classes_list = _prob_list[2]
            
        if DEBUG > 0 : print ('priorProbTable=%s'%(priorProbTable))
        if DEBUG > 0 : print ('posteriorProbTable=%s'%(posteriorProbTable))  
        if DEBUG > 0 : print ('classes_list=%s'%(classes_list))  
        
        return priorProbTable, posteriorProbTable, classes_list
    except IOError as _err:
        if (1): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()

def print_list(l):
    for i in l:
        print(i)
 

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

def get_tagging(documents, priorProbTable, posteriorProbTable, classes_list):
    tagged_line = ''
    tagged_document = []
    review, sentences = '', ''
    tokenize = tokenizer(STOP_WORDS, PUNCTUATION)
    data_column = TEST_COLUMNS -1
    nb = Naive_Bayes(priorProbTable, posteriorProbTable) 
    
    for _each_line in documents: #row is x
        review = _each_line.rstrip('\n').split(TOKEN_DELIMITER, data_column)

        review[data_column] = tokenize.get_wordlist(review[data_column], ascii_only=ASCII_ONLY, remove_stopwords=REMOVE_STOPWORDS, remove_punctuation = REMOVE_PUNCTUATION)
        sentences = review[data_column]
        if DEBUG > 0:
            print (review)
                
        if PRINT_TIME > 1: print ('nbclassify.decode(sentence)=>Start=>%s'%(str(datetime.now()))) 
        tagged_line = nb.classify(sentences, classes_list)
        if DEBUG > 0: print ('tagged_line=%s'%(tagged_line))
        tagged_document.append(review[0]+' '+tagged_line)
        
    return tagged_document

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


def evaluate(predicts, truth_file):
    predict = {}
    truth = {}
    y_predict_list = []
    y_truth_list = []
    
    try: 
        with open(truth_file, 'r', encoding='utf-8') as fp:
            for _each_line in fp:
                _each_line =_each_line.rstrip('\n').split(TOKEN_DELIMITER)
                truth[_each_line[0]]=[_each_line[1],_each_line[2]]
        fp.close()
                          
    except IOError as _err:
        if (DEBUG): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()
    
    for _each_predict in predicts:
        _each_predict = _each_predict.split(TOKEN_DELIMITER)
        predict[_each_predict[0]]=[_each_predict[1],_each_predict[2]]

    for key, val_pair in predict.items():
        for i, v in enumerate(val_pair):
            y_predict_list.append(v)
            y_truth_list.append(truth[key][i])
    classification_report(y_truth_list, y_predict_list)
#     ## you can also use the function from sklearn package.          
#     from sklearn import metrics
#     print('Result Report\n %s'%(metrics.classification_report(y_truth_list, y_predict_list, digits=4)))            

def classification_report(truth_list, predict_list):
    '''
    results = {class1:{TP:count, FP: count, FN:count}, ...}
    '''
    count_results = {}
    score_results = {}
    
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
        precision = result['TP']/(result['TP']+result['FP'])
        support = result['TP']+result['FN']
        recall = result['TP']/support
        F1 = 2*precision*recall/(precision+recall)
        score_results[key] = {'precision': precision, 'recall': recall, 'f1-score': F1, 'support': support}
        print ('class:%s, precision=%f, recall=%f, f1-score=%f, support=%d'%( key, precision, recall, F1, support ))
    total_f1 = 0
    total_support = 0
    for key, result in score_results.items():
        total_f1 += result['f1-score']*result['support']
        total_support += result['support']
    print ('average f1=%f'%(total_f1/total_support))
    return score_results   
    
    
            
'''
    Main program for the HMM decoder class execution.

'''
           
if __name__ == '__main__':
    '''
    Main program.
        1. Read the model file from MODEL_FILE_NAME = './nbmodel.txt' as default.
        2. Construct Naive Bayes model.
        3. Drop out unknown words.
    '''      
    # Get input and output parameters
    argv_len = len(sys.argv)
    if argv_len != 2 and argv_len != 3:
        print('Usage: ' + sys.argv[0] + ' /path/to/inputfile [/path/to/answerfile]')
        sys.exit(1)
    if PRINT_TIME : print ('nbclassify.get_input=>%s'%(str(datetime.now())))  
    # Assign the input and output variables
    INPUT_FILE = sys.argv[1]
    
    documents = get_input(INPUT_FILE)
    if PRINT_TIME : print ('nbclassify.load_parameters=>Start=>%s'%(str(datetime.now())))    
    priorProbTable, posteriorProbTable, classes_list = load_parameters(MODEL_FILE_NAME)    
    if PRINT_TIME : print ('nbclassify.get_tagging=>Start=>%s'%(str(datetime.now())))     
    tagged_document = get_tagging(documents, priorProbTable, posteriorProbTable, classes_list)
    if PRINT_TIME : print ('nbclassify.set_output=>Start=>%s'%(str(datetime.now())))      
    set_output(OUTPUT_FILE_NAME, tagged_document)
    if PRINT_TIME : print ('nbclassify.set_output=>end=>%s'%(str(datetime.now())))  
    
    if argv_len == 3:
        evaluate(tagged_document, sys.argv[2])
        
    