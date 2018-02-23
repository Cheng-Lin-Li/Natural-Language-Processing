#!/usr/bin/env python
# encoding: utf-8
'''
Natural Language Processing
Algorithm Name: Hidden Markov Model implementation on tagging process.

This is a program to implement of Hidden Markov Model part-of-speech tagger for English, Chinese.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2018 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    chenglil@usc.edu or clark.cl.li@gmail.com
@version:    1.1

@create:    February 15, 2018
@updated:   February 22, 2018

Tasks:
The program implements Hidden Markov Model part-of-speech tagger for English, Chinese, and a surprise language. 
The training data are provided in a tokenized and tagged file; the test data will be provided tokenized.
This program will build up transition probabilities table and emission probabilities table, saves in the same directory as hmmmodel.txt.
The other tagger program (hmmdecode3.py) will read the model file and a tokenized document to add the tags. 

The solution will use add-one smoothing on the transition probabilities and no smoothing on the emission probabilities; 
for unknown tokens in the test data it will ignore the emission probabilities and use the transition probabilities alone. 

Data:

A set of training and development data will be made available as separate files. 
1. Two files (one English, one Chinese) with tagged training data in the word/TAG format, with words separated by spaces and each sentence on a new line. 
2. Two files (one English, one Chinese) with untagged development data, with words separated by spaces and each sentence on a new line. 
3. Two files (one English, one Chinese) with tagged development data in the word/TAG format, with words separated by spaces and each sentence on a new line, to serve as an answer key. 

Data Structures and Global variables:
1. MODEL_FILE_NAME = './hmmmodel.txt'
2. TOKEN_DELIMITER = ' ' #the splitter for each token ( word/tag ).
3. TAG_DELIMITER ='/' #separate word and tag
4. INITIAL_STATE = 'Q' 
5. END_STATE = 'E'
6. CD_TAG_THRESHOLD = 20 # if a word appears more than the threshold with tag CD, then record it.
7. PATTERNS = Word feature pattern for CD, ADD, and NN 
        [    
            ('^[-|+]?[0-9]+(.[0-9]+)?', 'CD'),
            ('[-|+]?[0-9]+(.[0-9]+)?$', 'CD'),
            ('^[-|+]?[0-9]+(.[0-9]+)+(.[0-9]+)?$', 'CD'),            
            ('^([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)([\.a-zA-Z]{0,5})', 'ADD'),
            ('^(http\:\/\/|https\:\/\/)', 'ADD'),
            ('[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'ADD'),
            ('\.((?i)csv|jpe?g|gif|txt|doc|docx|ppt|pptx|xls|xlsx|pdf|mp3|mp4)$', 'NN')
        ]
8. NUMBER_PATTERN = '^[-|+]?[0-9]+(.[0-9]+)?' # Get the TAG of number from corpus.
'''
from __future__ import print_function 
from __future__ import division

import sys
import collections
import copy, json, re
from datetime import datetime

DEBUG = 0 # 1 = print debug information, 2=detail steps information 
PRINT_TIME = 0 # 0= disable, 1 = print time stamps, 2 = print detail time stamps
MODEL_FILE_NAME = './hmmmodel.txt'
TOKEN_DELIMITER = ' ' #the splitter for each token ( word/tag ).
TAG_DELIMITER ='/' #separate word and tag
INITIAL_STATE = 'Q' 
END_STATE = 'E'
CD_TAG_THRESHOLD = 20 # if a word appears more than the threshold with tag CD, then record it.
PATTERNS = [    
            ('^[-|+]?[0-9]+(.[0-9]+)?', 'CD'),
            ('[-|+]?[0-9]+(.[0-9]+)?$', 'CD'),
            ('^[-|+]?[0-9]+(.[0-9]+)+(.[0-9]+)?$', 'CD'),            
            ('^([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)([\.a-zA-Z]{0,5})', 'ADD'),
            ('^(http\:\/\/|https\:\/\/)', 'ADD'),
            ('[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'ADD'),
            ('\.((?i)csv|jpe?g|gif|txt|doc|docx|ppt|pptx|xls|xlsx|pdf|mp3|mp4)$', 'NN')
        ]
NUMBER_PATTERN = '^[-|+]?[0-9]+(.[0-9]+)?'

def get_input(file_name):
    document = []
    _tl = []
    _nd = []
    try: 
        with open(file_name, 'r', encoding='utf-8') as _fp:
            for _each_line in _fp:
                _each_line =_each_line.strip()
                document.append(_each_line)                    
        return document 
    except IOError as _err:
        if (DEBUG): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()

def print_list(l):
    for i in l:
        print(i)
    
    
def get_transit_probability(count_table):
    '''
    Convert number of counts to probabilities table.
        count_table is a dictionary structure
        Data structure: = OrderedDict([('NNP', {'HYPH': 171, ':': 51, 'NNP': 3036, ...}),()...])
    '''
    transition_prob = collections.OrderedDict() #transition probabilities table
      
    for _key, _val in count_table.items():
        total_count = 0
        for _k, _v in _val.items():
            total_count += _v # Sum the count
        for _k in _val.keys():
            #transition_prob[(_key, _k)] = _val[_k]/total_count
            _val[_k] = _val[_k]/total_count
    
    return count_table   

def get_emission_probability(states, count_table):
    '''
    Convert number of counts to probabilities table.
        count_table is a dictionary structure
        Data structure: = OrderedDict([('Al', {'NNP': 78}), ('-', {'HYPH': 643, ':': 254, ...}),...]) 
    ''' 
    for _word in count_table.keys():
        _item_dict = count_table.get(_word)
        for _k, _v in _item_dict.items(): # _k = tag
            total_count =  states[_k]# total counts for the state/tag
            _item_dict[_k] = _item_dict[_k]/total_count
    
    return count_table 




def get_probabilities_tables(document):
    '''
    Calculate the probability tables of initial, transition, and emission probabilities.
    
    Store states (POS tags) into variable "states" in a dictionary data structure.
        Data structure: states=OrderedDict([('NNP', 11834), ('HYPH', 664), (':', 696),....])
    Store initial probability into variable "initial_prob" in a dictionary data structure.
        Data structure: initial_prob=OrderedDict([('Q', {'NNP': 1156, '-LRB-': 138, 'CD': 407, ...}),()])
    observations store observation counts with each state
        Data structure observations = OrderedDict([('Al', {'NNP': 78}), ('-', {'HYPH': 643, ':': 254, ...}),...])    
    state_observation stores state with number of observations.
        Data structure 
            state_observation = {stateA: [total number of words which appear with stateA, number of words which appeared once with stateA], ...}
    Store transition probabilities into variable "transition_prob" in a dictionary data structure.
        Data structure: transition_prob=OrderedDict([('NNP', {'HYPH': 171, ':': 51, 'NNP': 3036, ...}),()])
    Store emission probabilities into variable "emission_prob" in a dictionary data structure.
        Data structure: mission_prob=OrderedDict([('Al', {'NNP': 78}), ('-', {'HYPH': 643, ':': 254, ...}),...])    
    '''
    
    states = collections.OrderedDict() # tag dictionary
    observations = collections.OrderedDict() # observation dictionary
    state_observation_summary_counts = collections.OrderedDict() # state with observation summary count dictionary for good turing smoothing
    # {stateA: [total number of words which appear with stateA, number of words which appeared once with stateA]}
    state_observation_counts = collections.OrderedDict() # state with detail observations and their count dictionary
    # {stateA: {'symbol1': counts, 'symbol2': counts,...}} Identify words for CD
    initial_prob = collections.OrderedDict() #initial probabilities table
    transition_prob = collections.OrderedDict() #transition probabilities table
    emission_prob = collections.OrderedDict() #emission probabilities table
    pattern_tag_dict = {}
    word, tag, previous_tag = '', '', ''
    
    
    # Match the state to x, y coordination.
    _tmp_state_obs_dict = {} 
    for _each_line in document: #row is x
        sentence = _each_line.rstrip('\n').split(TOKEN_DELIMITER)
        _i = 0
        for _each_token in sentence: # check every word/token
            word_tag = _each_token.rsplit(TAG_DELIMITER, 1)
            if DEBUG > 0: print (word_tag)

            if _i == 0 :
                # Initial state
                previous_tag = INITIAL_STATE # First word in every sentence
                word = word_tag[0]
                tag = word_tag[1]
                _tmp_dict = initial_prob.get(INITIAL_STATE, {tag: 0})
                _tmp_dict[tag] = _tmp_dict.get(tag, 0) + 1
                initial_prob[INITIAL_STATE] = _tmp_dict

                _i = 1
            else :
                # Transition & Emission table build up
                previous_tag = tag # Get previous tag from second word in each sentence
                word = word_tag[0]
                tag = word_tag[1]
                _tmp_dict = transition_prob.get(previous_tag, {tag: 0})
                _tmp_dict[tag] = _tmp_dict.get(tag, 0) + 1
                transition_prob[previous_tag] = _tmp_dict                
            
            # Try to get additional data for Good Turing smoothing and word features
            state_observation_summary_counts, state_observation_counts, _tmp_state_obs_dict, pattern_tag_dict =\
                get_smoothing_and_feature_data(_tmp_state_obs_dict, tag, word, state_observation_summary_counts, state_observation_counts, pattern_tag_dict)

            
            states[tag] = states.get(tag, 0)+1
            # get tag for each word. If no tag exist, return 0
            _tmp_dict = emission_prob.get(word, {tag: 0})
            _tmp_dict[tag] = _tmp_dict.get(tag, 0) + 1 # add counts into tag
            emission_prob[word] = _tmp_dict
        
        # Record the End state
        _tmp_dict = transition_prob.get(tag, {END_STATE: 0})
        _tmp_dict[END_STATE] = _tmp_dict.get(END_STATE, 0) + 1
        transition_prob[tag] = _tmp_dict 
    
    # Copy words counts from emission counting dictionary
    observations = copy.deepcopy(emission_prob) 
    #Smoothing for initial and transition probabilities.
    states_with_end_state = copy.deepcopy(states)
    states_with_end_state[END_STATE] = 1
    initial_prob = add_one_smoothing({INITIAL_STATE:1}, states, initial_prob)
    transition_prob = add_one_smoothing(states, states_with_end_state, transition_prob)
    pattern_tag_dict = get_max_tag_with_pattern(pattern_tag_dict)
    
    # unknown_words_features(state_observation_counts)
    number_words_dict = get_number_word(state_observation_counts, pattern_tag_dict.get(NUMBER_PATTERN))
           
    initial_prob = get_transit_probability(initial_prob)    
    transition_prob = get_transit_probability(transition_prob)
    emission_prob = get_emission_probability(states, emission_prob)

    if DEBUG > 0 : print ('states=%s'%(states))
    if DEBUG > 0 : print ('observations=%s'%(observations))
    if DEBUG > 0 : print ('state_observation_summary_counts=%s'%(state_observation_summary_counts))      
    if DEBUG > 0 : print ('state_observation_counts=%s'%(state_observation_counts))    
    if DEBUG > 0 : print ('_tmp_state_obs_dict=%s'%(_tmp_state_obs_dict))     
    
    if DEBUG > 0 : print ('initial_prob=%s'%(initial_prob))    
    if DEBUG > 0 : print ('transition_prob=%s'%(transition_prob))
    if DEBUG > 0 : print ('emission_prob=%s'%(emission_prob)) 
    if DEBUG > 0 : print ('number_words_dict=%s'%(number_words_dict))  
    if DEBUG > 0 : print ('pattern_tag_dict=%s'%(pattern_tag_dict))         
   
            
    #Save the model to MODEL_FILE_NAME
    with open(MODEL_FILE_NAME, 'w', encoding='utf-8') as fp:
        json.dump([states, observations, state_observation_summary_counts, initial_prob, transition_prob, emission_prob, \
                   number_words_dict, pattern_tag_dict], fp , indent=1, ensure_ascii=False)
        #json.dump([states, observations, initial_prob, transition_prob, emission_prob], fp , ensure_ascii=False)    
    return initial_prob, transition_prob, emission_prob

def get_max_tag_with_pattern(pattern_tag_dict):
    '''
    Input pattern_tag_dict includes word feature pattern and it's tag in corpus with count
        {'word feature pattern': {'tag1': counts, 'tag2': counts}}
    This function will return max. counts of tag for each word feature pattern
        {'word feature pattern': 'tag'}
    '''
    patt_tag = {}
    for k, v in pattern_tag_dict.items():
        _tag = max(v, key=lambda x: v[x])
        patt_tag[k] = _tag
    # Check existing pattern table, add not found patterns and tags
    
    for v in PATTERNS:
        if patt_tag.get(v[0], None) is None:
            patt_tag[v[0]] = v[1] #assign tag with pattern
    
    return patt_tag

def get_smoothing_and_feature_data(state_obs_dict, tag, word, state_observation_summary_counts, state_observation_counts, pattern_tag_dict):
    '''
    Additional data for Good-Turing smoothing and feature engineering.
    scan each word and tag to accumulate tag, word counting information.
    '''
    tag_count_dict = {}
    
    for pattern in PATTERNS: # try each word feature pattern
        _patt = pattern[0]
        result = re.search(pattern[0], word)
        if result:  # match.
            tag_count_dict = pattern_tag_dict.get(_patt, {tag: 0})
            tag_count_dict[tag] = tag_count_dict.get(tag, 0) +1 
            pattern_tag_dict[_patt] = tag_count_dict 
        else: # not match pattern                    
            continue
        
    if state_obs_dict.get(tag, {}).get(word, 0) == 0: # new word for current tag.
        _tmp_dict = state_obs_dict.get(tag, {})
        _tmp_dict[word] = 1
        state_obs_dict[tag]=_tmp_dict
        _tmp_list = state_observation_summary_counts.get(tag, [0, 0])
        state_observation_summary_counts[tag] = [_tmp_list[0]+1, _tmp_list[1]+1] #add 1 for total count and single word
        state_observation_counts[tag] = _tmp_dict
    elif state_obs_dict.get(tag).get(word, 0) == 1: # the word tagged with current more than once.
        _tmp_dict = state_obs_dict.get(tag, {})
        _tmp_dict[word] = _tmp_dict.get(word, 0) + 1
        state_obs_dict[tag]=_tmp_dict
        _tmp_list = state_observation_summary_counts.get(tag)
        state_observation_summary_counts[tag] = [_tmp_list[0]+1, _tmp_list[1]-1] #add 1 for total count of words
        state_observation_counts[tag] = _tmp_dict
    else: # the word tagged with current more than twice.
        _tmp_dict = {word: state_obs_dict.get(tag).get(word, 0) + 1} 
        _tmp_dict = state_obs_dict.get(tag, {})
        _tmp_dict[word] = _tmp_dict.get(word, 0) + 1
        state_obs_dict[tag]=_tmp_dict
        _tmp_list = state_observation_summary_counts.get(tag)
        state_observation_summary_counts[tag] = [_tmp_list[0]+1, _tmp_list[1]] #add 1 for total count of words
        state_observation_counts[tag] = _tmp_dict
        
    
    return state_observation_summary_counts, state_observation_counts, state_obs_dict, pattern_tag_dict

def get_number_word(state_observation_counts, number_tag):
    '''
        Identify which word/symbol/observation is related to number
    '''
    number_word_list = []
    #get all words with number tag. It should be "CD" by default but may be different from tagged training set
    cd_dict = state_observation_counts.get(number_tag, {}) 
    for k, v in cd_dict.items() :
        if not k.isdigit() and v > CD_TAG_THRESHOLD: #The word is not digits and appears more than threshold.
            if len(k) == 1 and re.match(r'^[a-zA-Z]$', k): # Ignore single alphabet
                continue
            else:
                number_word_list.append(k) 
    
    return {number_tag: number_word_list}
def add_one_smoothing(previous_states, current_states, prob_table):
    '''
        apply add one smoothing on the transition probability table
    '''
    current_states_number = len(current_states)
    
    for _pkey in previous_states.keys():
        _current_states = prob_table.get(_pkey, {})
        # If some state transition not exist, then apply add one smoothing        
        if len (_current_states) < current_states_number:
            for _ckey in current_states.keys():
                _current_states[_ckey] = _current_states.get(_ckey, 0) + 1
            prob_table[_pkey] = _current_states
             
    return prob_table
    
'''
    Main program for the initial, transit, and emission probabilities tables generate for HMM class execution.

'''
           
if __name__ == '__main__':
    '''
    Main program.
        1. Read the training/corpus file from en_train_tagged.txt as default.
        2. Construct HMM transfer probability table by POS tags, and emission probability table by observation word.
        3. Store these two tables for hmmdecode3.py to read and perform tagger tasks.
    '''      
    # Get input and output parameters
    if len(sys.argv) != 2:
        print('Usage: ' + sys.argv[0] + ' /path/to/inputfile ')
        sys.exit(1)
    
    # Assign the input and output variables
    INPUT_FILE = sys.argv[1]
    if PRINT_TIME : print ('hmmlearn.get_input=>Start=>%s'%(str(datetime.now())))    
    document = get_input(INPUT_FILE)
    
    ##################

    if DEBUG > 0: print_list(document)
    if PRINT_TIME : print ('hmmlearn.get_probabilities_tables=>Start=>%s'%(str(datetime.now())))    
    initialProbTable, transitionProbTable, emissionProbTable = get_probabilities_tables(document)
    if PRINT_TIME : print ('hmmlearn.get_probabilities_tables=>End=>%s'%(str(datetime.now())))    
    
        