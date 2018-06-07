#!/usr/local/bin/python3.6
# encoding: utf-8
'''
Perceptron.percepclassify -- This perceptron classifiers  include vanilla and averaged models 
will read model parameters from file then perform classification tasks.
 

Perceptron.percepclassify is a perceptron classifiers which include vanilla and averaged models 
will read model parameters from file then perform classification tasks on 
identify hotel reviews as either true or fake, and either positive or negative. 
The word tokens will be treated as features, or other features may devise from the text. 

The program will read perceptron models, and perform classification tasks. 
The model parameter may indicate to file: 
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

Reference F1 score: 0.88 for vanilla perceptron and 0.89 for the averaged perceptron,
'''
from __future__ import print_function 
from __future__ import division

__all__ = []
__version__ = 0.1
__date__ = '2018-04-18'
__updated__ = '2018-04-18'

import sys, os
import collections
import math, json, re
from datetime import datetime
import numpy as np


DEBUG = 0 # 1 = print debug information, 2=detail steps information 
PRINT_TIME = 0 # 0= disable, 1 = print time stamps, 2 = print detail time stamps
VANILLA_MODEL_FILE_NAME = './vanillamodel.txt'
AVERAGE_MODEL_FILE_NAME = './averagedmodel.txt'
TOKEN_DELIMITER = ' ' #the splitter for each token ( word/tag ).
TEST_COLUMNS = 2
LOW_FREQ_OBSERVATION_THRESHOLD = 2 # words appear more than or equal to the number of times in any one class will be reserved
HIGH_FREQ_OBSERVATION_THRESHOLD = 1000 # words appear less than or equal to the number of times in any one class will be reserved

ASCII_ONLY = True
REMOVE_STOPWORDS = True
REMOVE_PUNCTUATION = True

# Stanford NLP + NLTK stop words
STOP_WORDS = []
PUNCTUATION = []
OUTPUT_FILE_NAME = './percepoutput.txt'

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
    word_dict = {}
    parameters_list = []
    classes_list = []
    
    try: 
        #Load the model from MODEL_FILE_NAME
        with open(file_name, 'r', encoding='utf-8') as fp:
            p_list = json.load(fp)

            word_dict =  p_list[0]
            parameters_list = p_list[1]
            classes_list = p_list[2]

        if DEBUG > 0 : print ('word_dict=%s'%(word_dict))            
        if DEBUG > 0 : print ('parameters_list=%s'%(parameters_list))
        if DEBUG > 0 : print ('classes_list=%s'%(classes_list))   
        
        return word_dict, parameters_list, classes_list
    except IOError as _err:
        if (1): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()

def print_list(l):
    for i in l:
        print(i)
 
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
                        
    def vanilla_perceptron(self, features, labels):
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
        
        X = features
        Y = labels

        weights = np.zeros(shape=(1, features.shape[1])) # Get the number of vocabularies
        bias = 0
        for _it in range(self.iteration):
            for i in range(len(X)):
                x = X[i]
                y = np.array([Y[i]])
                _a = np.dot(weights, x.transpose()) + bias
                if y*_a <= 0:
                    weights = weights + np.dot(y, x)
                    bias = bias + y
                if DEBUG > 0 : print('iteration:%d, y=%f, _a=%f, x=%s, b=%f, w=%s'%(_it, y, _a, str(x), bias, str(weights)))
        
        self.weights = weights
        self.bias = bias
        
        return [weights.tolist(), bias.tolist()]
    
    def averaged_perceptron(self, features, labels):
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
        X = features
        Y = labels

        weights = np.zeros(shape=(1, features.shape[1])) # Get the number of vocabularies
        bias = 0

        u = np.zeros(shape=(1, features.shape[1])) # Get the number of vocabularies
        b = 0
        c = 1
                
        for _it in range(self.iteration):
            for i in range(len(X)):
                x = X[i]
                y = np.array([Y[i]])
                _a = np.dot(weights, x.transpose()) + bias
                if y*_a <= 0:
                    weights = weights + np.dot(y, x)
                    bias = bias + y
                    
                    u = u + np.dot(y*c, x)
                    b = b + y*c
                c += 1
                if DEBUG > 0 : ('iteration:%d, y=%f, _a=%f, x=%s, b=%f, w=%s'%(_it, y, _a, str(x), bias, str(weights)))
        
        self.weights = weights-(1/c)*u
        self.bias = bias-(1/c)*b
        
        return [self.weights.tolist(), self.bias.tolist()]

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
        
class doc2vec(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def get_vector(self, data):
        dictionary = self.dictionary

        vector = np.zeros(shape=(1, len(dictionary)))
        for element in data:
            _idx = dictionary.get(element, -1)
            if _idx != -1: # Get vocabulatory from dictionary
                vector[0][_idx] += 1
            else: # skip unseen words
                pass
     
        return vector
    
def get_tagging(documents, word_dict, parameters_list, classes_list):
    tagged_line = ''
    tagged_document = []
    review, sentences = '', ''
    tokenize = tokenizer(STOP_WORDS, PUNCTUATION)
    data_column = TEST_COLUMNS -1
    reverse_classes_list = []
    
    for class_list in classes_list:
        _tmp_dict = {}
        for k,v in class_list.items():
            _tmp_dict[v] = k
        reverse_classes_list.append(_tmp_dict)                            
    # Perceptron True or Fake
    p_tf = Perceptron() 
    p_tf.load_model(parameters_list[0][0], parameters_list[0][1])

    # Perceptron Positive or Negative
    p_pn = Perceptron() 
    p_pn.load_model(parameters_list[1][0], parameters_list[1][1])    
    d2v = doc2vec(word_dict)
    
    for _each_line in documents: #row is x
        review = _each_line.rstrip('\n').split(TOKEN_DELIMITER, data_column)

        review[data_column] = tokenize.get_wordlist(review[data_column], ascii_only=ASCII_ONLY, remove_stopwords=REMOVE_STOPWORDS, remove_punctuation = REMOVE_PUNCTUATION)
        
        sentences = d2v.get_vector(review[data_column])
        if DEBUG > 0:
            print (review)
                
        tagged_line_tf = p_tf.predict(sentences, reverse_classes_list[0])
        tagged_line_pn = p_pn.predict(sentences, reverse_classes_list[1])
        if DEBUG > 0: print ('predicted_review: True or Fake=%s, Positive or Negative=%s'%(tagged_line_tf[0], tagged_line_pn[0]))
        tagged_document.append(review[0]+' '+tagged_line_tf[0]+' '+tagged_line_pn[0])
        
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

def classification_report(truth_list, predict_list, print_results=True):
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
    
def main(input_doc, model, answer=None):
    parameters_list = [] #[class1[weights, bias], class2[weights, bias]]
        
    if PRINT_TIME : print ('percepclassify.get_input=>Start=>%s'%(str(datetime.now())))
    documents = get_input(input_doc)
    
    if PRINT_TIME : print ('percepclassify.load_parameters=>Start=>%s'%(str(datetime.now())))   
 
    word_dict, parameters_list, classes_list = load_parameters(model)   
      
    if PRINT_TIME : print ('percepclassify.get_tagging=>Start=>%s'%(str(datetime.now())))   
    tagged_document = get_tagging(documents, word_dict, parameters_list, classes_list)
    if PRINT_TIME : print ('percepclassify.set_output=>Start=>%s'%(str(datetime.now())))      
    set_output(OUTPUT_FILE_NAME, tagged_document)
    if PRINT_TIME : print ('percepclassify.set_output=>end=>%s'%(str(datetime.now())))  
    
    if answer != None:
        evaluate(tagged_document, answer)    
            
'''
    Main program for the HMM decoder class execution.

'''
           
if __name__ == '__main__':
    '''
    Main program.
        1. Read the model file from MODEL_FILE_NAME = './vanillamodel.txt or averagedmodel.txt' for different models.
        2. Construct Perceptron algorithm to vanilla model or averaged model for two classification tasks.
        3. Drop out unknown words.
        4. Predict each classification on each review.

    '''      
    # Get input and output parameters
    argv_len = len(sys.argv)
    if argv_len != 3 and argv_len != 4:
        print('Usage: ' + sys.argv[0] + ' /path/to/model_file /path/to/inputfile [/path/to/answerfile]')
        sys.exit(1)

    # Assign the input and output variables
    MODEL_FILE_NAME = sys.argv[1]
    INPUT_FILE = sys.argv[2]
    if argv_len == 4:
        ANSWER_FILE = sys.argv[3]
    else:
        ANSWER_FILE = None
    main (INPUT_FILE, MODEL_FILE_NAME, ANSWER_FILE)    
#     main (INPUT_FILE, VANILLA_MODEL_FILE_NAME, ANSWER_FILE)  
#     main (INPUT_FILE, AVERAGE_MODEL_FILE_NAME, ANSWER_FILE)  