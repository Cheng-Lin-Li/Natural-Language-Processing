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
6. PATTERNS = [] #read word feature patterns from file
7. NUMBER_LIST = [] # candidate word list which related to number.
8. THESHOLD_NUMBER = 2 # How many number candidate words appear in a token/symbol will be considered a number.
'''
from __future__ import print_function 
from __future__ import division

import sys
import collections
import math, copy, json, re
from datetime import datetime


DEBUG = 0 # 1 = print debug information, 2=detail steps information 
PRINT_TIME = 0 # 0= disable, 1 = print time stamps, 2 = print detail time stamps
MODEL_FILE_NAME = './hmmmodel.txt'
TOKEN_DELIMITER = ' ' #the splitter for each token ( word/tag ).
TAG_DELIMITER ='/' #separate word and tag
INITIAL_STATE = 'Q' 
END_STATE = 'E'
OUTPUT_FILE_NAME = './hmmoutput.txt'
PATTERNS = [] #read word feature patterns from file
NUMBER_LIST = [] # candidate word list which related to number.
THESHOLD_NUMBER = 2 # How many number candidate words appear in a token/symbol will be considered a number. 

def get_input(file_name):
    document = []

    try: 
        with open(file_name, 'r', encoding='utf-8') as fp:
            for _each_line in fp:
                _each_line =_each_line.rstrip('\n').split(TOKEN_DELIMITER)
                document.append(_each_line)  
        fp.close()                  
        return document 
    except IOError as _err:
        if (DEBUG): 
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
        if (DEBUG): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()        
        
def load_parameters(file_name):
    # Load initial, transit, and emission probabilities tables generate for HMM class execution.
    states = collections.OrderedDict()
    observations = collections.OrderedDict()
    state_observation_counts = collections.OrderedDict()
    initial_prob = collections.OrderedDict()
    transition_prob = collections.OrderedDict()
    emission_prob = collections.OrderedDict()
    number_words_dict = {}
    pattern_tag_dict = {}
    
    try: 
        #Load the model from MODEL_FILE_NAME
        with open(file_name, 'r', encoding='utf-8') as fp:
            _prob_list = json.load(fp)
            states = _prob_list[0]
            observations = _prob_list[1]
            state_observation_counts = _prob_list[2]
            initial_prob = _prob_list[3]
            transition_prob = _prob_list[4]
            emission_prob = _prob_list[5]
            number_words_dict = _prob_list[6]
            pattern_tag_dict = _prob_list[7]
            
        if DEBUG > 0 : print ('states=%s'%(states))
        if DEBUG > 0 : print ('observations=%s'%(observations))  
        if DEBUG > 0 : print ('state_observation_counts=%s'%(state_observation_counts))  
        if DEBUG > 0 : print ('initial_prob=%s'%(initial_prob))    
        if DEBUG > 0 : print ('transition_prob=%s'%(transition_prob))
        if DEBUG > 0 : print ('emission_prob=%s'%(emission_prob)) 
        if DEBUG > 0 : print ('number_words_dict=%s'%(number_words_dict))         
        
        # Get default word feature pattern
        for k, v in pattern_tag_dict.items():
            PATTERNS.append((k, v))
            
        for k, v_list in number_words_dict.items():
            NUMBER_LIST.extend((k, v_list)) 
          
        if DEBUG > 0: print ('additional NUMBER_LIST=%s'%(NUMBER_LIST))
        return states, observations, state_observation_counts, initial_prob, transition_prob, emission_prob
    except IOError as _err:
        if (DEBUG): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()

def print_list(l):
    for i in l:
        print(i)
 

class HMM(object):
    '''
    Hidden Markov Model (HMM) apply to tokenized document tagging.
    2. def eliminate_evidences(self, observation_seq, step): To support multiple evidences      
        Input: observation sequence of evidences, what is the time step
        Return the probability of combination of all evidences.
    3. def max_prob_state(self, states_prob): To get maximum probability of state list at time t
        Input: current states of probability which constrain by the probability of previous states
        Return: max_prob list for each state at time t and its previous state at time t-1    


    '''

    def __init__(self, states, observations, init_prob=None, transition_prob=None, emission_prob=None, state_observation_counts = None, smoothing = None, smoothing_parameter = None):
        self.states = states
        self.states_len = len(states)
        self.observation_counts = observations
        self.observation_seq = None
        self.different_observations = len(observations)
        self.state_observation_counts = state_observation_counts
        self.init_prob = init_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.previous_possible_tags = transition_prob.keys()
        self.trace_back_prob = [] #record down highest probability at each time steps
        self.trace_back_states = []
        self.smoothing_method = smoothing
        self.unknown_words = smoothing_parameter # number of unknown words.
        self.lunda = smoothing_parameter
        self.patterns = PATTERNS
        self.decode = self.viterbi  
        
        if smoothing == 'lidstone':
            self.smoothing = self.lidstone_smoothing
        elif smoothing == 'good-turing':
            self.smoothing = self.good_turing_smoothing
        else:
            self.smoothing = self.no_smoothing
        
    def viterbi(self, observation_seq):            
        '''
        This is an implementation of Viterbi algorithm for tagger based on -log probability.
        '''
        states = copy.deepcopy(self.states)
        self.observation_seq = observation_seq
        current_state = ''
        best_score={(0, INITIAL_STATE): 0} #best score in each step
        best_edge={(0, INITIAL_STATE): None} #best edge in each step
        previous_possible_tags = [] #
        _tmp_previous_possible_tags = []
        state_prob = []
        current_state_prob = 0
        states_seq = '' # The out put of state sequences.            
        _obs = ''
        
        time_steps = len(observation_seq)
        if (time_steps == 0 or len(self.init_prob) == 0 ): #If input observation sequence is zero length.
            return states_seq
        else:
            pass
        if PRINT_TIME > 1 : print ('hmm.Forward Step=>start=>%s'%(str(datetime.now())))
                          
        #Forward Step
        for step in range(time_steps + 1): # Include additional end state.    
            if step == 0: #First step
                previous_possible_tags = {INITIAL_STATE: 1}
                current_possible_tags = self.init_prob.get(INITIAL_STATE)                
#                 current_possible_tags = states
                state_prob = self.init_prob         
            elif step == time_steps: # End step
                previous_possible_tags = _tmp_previous_possible_tags
                #previous_possible_tags = self.previous_possible_tags
                # Create previous states to  end state.
                current_possible_tags = {END_STATE: 1}
                state_prob = self.transition_prob
            else:
                previous_possible_tags = _tmp_previous_possible_tags
                #previous_possible_tags = self.previous_possible_tags
                current_possible_tags = states
                state_prob = self.transition_prob

            for previous_state in previous_possible_tags:
                current_state_prob = 0
                states_seq = ''
                _tmp_previous_possible_tags = []
                for current_state in current_possible_tags:
#                     # make sure the initial and transition are smoothing for all states. 
#                     current_state_prob = state_prob.get(previous_state).get(current_state, None)                    
                    
#                     if best_score.get((step, previous_state), None) is not None and current_state_prob is not None:
                    if best_score.get((step, previous_state), None) is not None:
                        if step == time_steps: # End step
                            emission_prob = 1
                        else:
                            _obs = observation_seq[step]
                            emission_prob = self.get_emission_prob(current_state, _obs)   
                            #If the emission probability is 0, then try next state.
                            if (emission_prob == 0): 
                                continue # Escape for current state
                        # make sure the initial and transition are smoothing for all states. 
                        current_state_prob = state_prob.get(previous_state).get(current_state, None)
                        _tmp_previous_possible_tags.append(current_state)
                        if DEBUG > 0 and _obs == 'ordinary':
                            if step == time_steps:                 
                                print ('previous_state=%s, current_state=%s, obs=E'%(previous_state, current_state))
                            else:
                                print ('current_state_prob=%s, emission_prob=%s, previous_state=%s, current_state=%s, obs=%s'% \
                                        (str(current_state_prob), str(emission_prob), str(previous_state), str(current_state), \
                                        str(_obs)))             

                        score = best_score.get((step, previous_state)) -math.log(current_state_prob) - math.log(emission_prob)
                        
                        if best_score.get((step+1, current_state), None) is None or best_score.get((step+1, current_state), None) > score:
                            best_score[(step+1, current_state)] = score
                            best_edge[(step+1, current_state)] = (step, previous_state)
                        else:
                            pass
                        
        if PRINT_TIME > 1 : print ('hmm.Backward Step:=>start=>%s'%(str(datetime.now())))          
        #Backward Step: back track to each time steps for the highest probability of state in each time step.
        next_edge = best_edge.get((time_steps+1, END_STATE))
        while (next_edge != (0, INITIAL_STATE)):
            _position, state = next_edge[0], next_edge[1]
            for pattern in self.patterns:
                result = re.search(pattern[0], observation_seq[_position-1])
                if result :
                    state = pattern[1]            
            #states_seq = f'{obs_seq[_position-1]}/{state} {states_seq}'
            states_seq = observation_seq[_position-1]+'/'+state+' '+states_seq # Reverse the sequence
            next_edge = best_edge.get(next_edge)
        
        return states_seq.rstrip(' ')

    
    def get_emission_prob(self, current_state, observation):
        '''
        To support emission probability based on current state
        In word tagging tasks, based on current observation and current state to estimate the probability of this combination.
            Apply smoothing algorithm if required.
        '''
        state_counts = self.observation_counts.get(observation, None)
        if state_counts is not None: 
            # Return the emission probability based on current state. 
            # If the symbol/word never found with current state in training set, return 0.
#             return state_counts.get(current_state, 0)
            return self.emission_prob.get(observation, {}).get(current_state, 0)
        else: #unknown words            
            # find unknow observation/symbol/word from pattern
            for pattern in self.patterns: # try each word feature pattern
                result = re.search(pattern[0], observation)
                if result is None:
                    continue # try next pattern
                elif pattern[1] == current_state: #unknown word match the pattern and tag 
                    if DEBUG > 1: print('get_emission_prob: observation=%s, current state=%s'%(observation, current_state))
                    return 1
                else: #unknown word match the pattern but different tag 
                    return 0
#             # find number pattern
            counter = 0
                   
            for n in NUMBER_LIST[1]:
                if n in observation:
                    counter += 1
            if counter >= THESHOLD_NUMBER :
                if current_state == NUMBER_LIST[0]: # if current tag/state is number
                    return 1
                else:
                    return 0
            else:
                #unknown words without pattern match
                return self.smoothing(current_state, observation)
        return self.smoothing(current_state, observation)

        
    
    def lidstone_smoothing(self, current_state, observation):
        '''
         bi(ot) = (Count(ot, qi)+1)/(Count(qi)+ Count(V))
         
             bi(ot) =  A word o (at time t) being emitted given a state qi. Probability of the symbol/word on state qi.
             Count(ot, qi) = The number of the symbol/word o (at time t) being emitted given a state qi
             Count(qi) = the number of state qi in the training text.
             lunda = 1: Laplace smoothing, Lunda < 1:  Lidstone’s law 
             Count(V) = V is the vocabulary size . The number of symbols and therefore the number of different words forms encountered in the training text.
        '''
        count_otqi = self.observation_counts.get(observation, {}).get(current_state, 0)
        count_qi = self.states.get(current_state)
        count_V = self.different_observations
        return (count_otqi+self.lunda)/(count_qi+self.lunda*count_V)        

    def good_turing_smoothing(self, current_state, observation=None):
        '''
         P(u|t) = n1(t)/(n0 * N(t))
         
             P(u|t): estimate Prob. for unknown word u with tag t.
             n0: number of unknown words.
             n1(t): number of words which appeared once with tag t.
             N(t): total number of words which appeared with tag t.
         '''
        unknown_words = 0
        if self.unknown_words is None: # If number of unknown words in whole document is not exists.
            unknown_words = 0
            for v in self.observation_seq: # Count number of unknown words in this sentence
                if self.observation_counts.get(v, 0) == 0:
                    unknown_words += 1
        else:
            unknown_words = self.unknown_words
                        
        n0 = unknown_words
        n1t = self.state_observation_counts.get(current_state)[1]
        Nt = self.state_observation_counts.get(current_state)[0]

        return (n1t)/(n0*Nt)

        
    def no_smoothing(self, current_state, observation=None):
        '''
         bi(ot) = 1         
             bi(ot) =  A word o (at time t) being emitted given a state qi. Probability of the symbol/word on state qi.
        '''
        return 1


def count_unknown_words(document, observation_counts):
    unknown_words = 0
    for _each_line in document:    
        for obs in _each_line: # Count number of unknown words
            if observation_counts.get(obs, 0) == 0:
                unknown_words += 1
    return unknown_words

def get_tagging(document, states, observations, state_observation_counts, initialProbTable, transitionProbTable, emissionProbTable):
    sentence = []
    tagged_line = ''
    tagged_document = []
    
    #unknown_words = count_unknown_words(document, observations)
    #hmm = HMM(states, observations, initialProbTable, transitionProbTable, emissionProbTable, state_observation_counts, 'good-turing', unknown_words)
    hmm = HMM(states, observations, initialProbTable, transitionProbTable, emissionProbTable, state_observation_counts, 'good-turing')
    #hmm = HMM(states, observations, initialProbTable, transitionProbTable, emissionProbTable, state_observation_counts, 'lidstone', 1) #last parameter must <= 1    
    #hmm = HMM(states, observations, initialProbTable, transitionProbTable, emissionProbTable, state_observation_counts)

    for _each_line in document:
        if PRINT_TIME > 1: print ('hmm.decode(sentence)=>Start=>%s'%(str(datetime.now()))) 
        sentence = _each_line 
        #unmark below line if input file is training data
#         sentence = [ x.rsplit(TAG_DELIMITER, 1)[0] for x in _each_line]

        tagged_line = hmm.decode(sentence)
        if DEBUG > 1: print (tagged_line)
        
        tagged_document.append(tagged_line)
    return tagged_document

def check_answer(predicts, answer_file):
    predict = []
    answer = []
    correct = 0
    error = 0

    try: 
        with open(answer_file, 'r', encoding='utf-8') as fp:
            for _each_line in fp:
                _each_line =_each_line.rstrip('\n').split(TOKEN_DELIMITER)
                answer.append(_each_line)  
        fp.close()
                          
    except IOError as _err:
        if (DEBUG): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()
    
    for _each_predict in predicts:
        _each_predict = _each_predict.split(TOKEN_DELIMITER)
        predict.append(_each_predict)
        
    for i, val_line in enumerate(predict):
        for j, val in enumerate(val_line):
            if (val == answer[i][j]):
                correct += 1
            else:
                error += 1
                if DEBUG > 0: print ('error=>predict="%s" , answer="%s"'%(val, answer[i][j]))
                
                
            
    print ('Correct = {}, Error = {}, Accuracy = {}%'.format(correct, error, correct/(error+correct)*100))              
    
        
'''
    Main program for the HMM decoder class execution.

'''
           
if __name__ == '__main__':
    '''
    Main program.
        1. Read the model file from MODEL_FILE_NAME = './hmmmodel.txt' as default.
        2. Construct HMM model.
        3. Apply Viterbi algorithm with specific smoothing method for unknown words.
        4. Smoothing methods: Good-Turing, lidstone_smoothing, no smoothing (Prob. = 1 for unknown words)
    '''      
    # Get input and output parameters
    argv_len = len(sys.argv)
    if argv_len != 2 and argv_len != 3:
        print('Usage: ' + sys.argv[0] + ' /path/to/inputfile [/path/to/answerfile]')
        sys.exit(1)
    if PRINT_TIME : print ('hmm.get_input=>%s'%(str(datetime.now())))  
    # Assign the input and output variables
    INPUT_FILE = sys.argv[1]
    
    document = get_input(INPUT_FILE)
    if PRINT_TIME : print ('hmmcode.load_parameters=>Start=>%s'%(str(datetime.now())))    
    states, observations, state_observation_counts, initialProbTable, transitionProbTable, emissionProbTable = load_parameters(MODEL_FILE_NAME)    
    if PRINT_TIME : print ('hmmcode.get_tagging=>Start=>%s'%(str(datetime.now())))     
    tagged_document = get_tagging(document, states, observations, state_observation_counts, initialProbTable, transitionProbTable, emissionProbTable)
    if PRINT_TIME : print ('hmmcode.set_output=>Start=>%s'%(str(datetime.now())))      
    set_output(OUTPUT_FILE_NAME, tagged_document)
    if PRINT_TIME : print ('hmmcode.set_output=>end=>%s'%(str(datetime.now())))  
    
    if argv_len == 3:
        check_answer(tagged_document, sys.argv[2])
        
    