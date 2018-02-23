# This is an implementation of Hidden Markov Model (HMM) - Viterbi Algorithm in Python 3

## Machine Learning Algorithm: Hidden Markov Model - Viterbi Algorithm.
## Smoothing methods: Add one, Good-Turing, lidstone_smoothing, no smoothing (Prob. = 1 for unknown words)

## Task:
Implementing a Hidden Markov Model part-of-speech tagger for English, Chinese, and any surprise language. The training data are provided tokenized and tagged; the test data will be provided tokenized, and the program / tagger will add the tags.

## Data:

A set of training and development data are available in the same folder with the following files: 
1. Two files (one English, one Chinese) with tagged training data in the word/TAG format, with words separated by spaces and each sentence on a new line. 
2. Two files (one English, one Chinese) with untagged development data, with words separated by spaces and each sentence on a new line. 
3. Two files (one English, one Chinese) with tagged development data in the word/TAG format, with words separated by spaces and each sentence on a new line, to serve as an answer key. 
4. A readme/license file

## Programs:
There are two programs: hmmlearn3.py will learn a hidden Markov model from the training data, and hmmdecode3.py will use the model to tag new data. The learning program will be invoked in the following way: 
```
> python hmmlearn3.py /path/to/input 
```
The argument is a single file containing the training data; the program will learn a hidden Markov model, and write the model parameters to a file called hmmmodel.txt. The format of the model is JSON, and the model file should contain sufficient information for hmmdecode.py to successfully tag new data. 
The model file should be human-readable, so that model parameters can be easily understood by visual inspection of the file. 


The tagging program will be invoked in the following way: 
```
> python hmmdecode3.py /path/to/input 
```
The argument is a single file containing the test data; the program will read the parameters of a hidden Markov model from the file hmmmodel.txt, tag each word in the test data, and write the results to a text file called hmmoutput.txt in the same format as the training data. 


## Usage: 
Training:
```
> python hmmlearn3.py /path/to/input/training_corpus 
```

Tagging/Decoding:
```
> python hmmdecode3.py /path/to/input/target_tokenized_file 
```

#### Input: 
1. text files 'en_train_tagged.txt', 'zh_train_tagged.txt' are training corpus for English and Chinese.
2. text files 'en_dev_raw.txt', 'zh_dev_raw.txt' are target files which need to be tagged for English and Chinese.
3. model file 'hmmmodel.txt' will automatically generate and read by programs.

#### Output: Output a tagged file with name hmmoutput.txt in the same folder of the program.


## Notes
1. Tags: Each language has a different tagset; the surprise language may have some tags that do not exist in the English and Chinese data. The program will build tag sets from the training data, and not rely on a precompiled list of tags. 
2. Slash character. The slash character ‘/’ is the separator between words and tags, but it also appears within words in the text. slashes never appear in the tags, so the separator is always the last slash in the word/tag sequence. 
3. Smoothing and unseen words and transitions. The program implements Good-Turing smoothing (lidstone_smoothing, no smoothing approach are options) to handle unknown vocabulary and add one smoothing methods for unseen transitions in the test data. 
  3-1. Unseen words: The test data may contain words that have never been encountered in the training data: these will have an emission probability of zero for all tags. 
  3-2. Unseen transitions: The test data may contain two adjacent unambiguous words (that is, words that can only have one part-of-speech tag), but the transition between these tags was never seen in the training data, so it has a probability of zero; in this case the Viterbi algorithm will have no way to proceed. 
4. The implementation use add-one smoothing on the transition probabilities and no smoothing on the emission probabilities; for unknown tokens in the test data it will ignore the emission probabilities and use the transition probabilities alone if you choose no smoothing option. The program default use Good-Turing method for unknown tokens. 
5. End state. The program implement the algorithm with transitions to an end state at the end of a sentence (as in Jurafsky and Martin, figure 9.11). 
4. Runtime efficiency. The program use log probabilities to contribute runtime-efficient code and prevent underflow. Path puring approach is also implemented to cut-off unnecessary probability calculations. 

