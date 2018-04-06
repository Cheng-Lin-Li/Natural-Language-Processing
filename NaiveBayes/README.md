# This is an implementation of Naive Bayes Classifier (NB) in Python 3

## Machine Learning Algorithm: Naive Bayes Classifier Algorithm.
## Smoothing methods: Laplace smoothing (drop out unseen words on testing data)

## Task:
Implementing a program to leverage Naive Bayes classifier model for sentiment reviews of English.

This task will write a naive Bayes classifier to identify hotel reviews as either true (True) or fake(Fake), and either positive(Pos) or negative(Neg). 

The program (nblearn3.py) will perform document split to the word level token as features to train for classification learning and generate probabilities tables into file. 

The argument is a single file containing the training data; the program will learn a naive Bayes model, 
and write the model parameters to a file called nbmodel.txt.

## Smoothing:
The solution uses add-one smoothing on the training data, and simply ignores unknown tokens in the test data.

## Tokenization: 
The program splits each word as a basic token unit. Certain punctuation will be removed, and lower casing all the letters. 
Stop word will be ignored. (high-frequency or low-frequency tokens). 


## Data:

A set of training and development data will be made available as separate files. 
1. The file (train-labeled.txt) is training data which include classification and reviews.
2. The file (dev-text.txt) is development set of data, with sequence no. and reviews which is separated by a new line. 
3. The file (dev-key.txt) is classifications for development set of data.
4. A readme/license file

## Programs:
There are two programs: nblearn3.py will learn probabilities model from the training data, and the classifier program (nbclassify3.py) will read the model file and reviews to perform classifications. The learning program will be invoked in the following way: 
```
> python nblearn3.py /path/to/input 
```
The argument is a single file containing the training data; the program will learn probabilities models, and write the model parameters to a file called nbmodel.txt. The format of the model is JSON, and the model file should contain sufficient information for nbclassify3.py to successfully classify new data. 
The model file is human-readable, so that model parameters can be easily understood by visual inspection of the file. 


The classifier program will be invoked in the following way: 
```
> python nbclassify3.py /path/to/input 
```
The argument is a single file containing the test data; the program will read the parameters of a Naive Bayes model from the file nbmodel.txt, classify each review in the test data, and write the results to a text file called nboutput.txt in the same format as the training data. 


## Usage: 
Training:
```
> python nblearn3.py /path/to/input/training_corpus 
```

Classification:
```
> python nbclassify3.py /path/to/input/target_tokenized_file 
```

#### Input: 
1. text files 'train-labeled.txt' is a training corpus for English.
2. text files 'dev-text' is a test file which need to be classify for English.
3. text files 'dev-key.txt' is a answer/label file for test file.
4. model file 'nbmodel.txt' will automatically generate and read by programs.

#### Output: Output a file with name nboutput.txt in the same folder of the program.


