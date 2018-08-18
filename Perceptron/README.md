# This is an implementation of Perceptron in Python 3

## Machine Learning Algorithm: Perceptron Algorithms: Vanilla and Averaged models.
## Smoothing method: Drop out unseen words on testing data
## Tuning methods: sequence, random sampling on training set, K-folds training.

## Task:
Implementing a program to leverage Perceptron binary classifier model for sentiment reviews of English by Vanilla and Averaged models.

For each perceptron model will identify hotel reviews on two classifications tasks: as either true (True) or fake(Fake), and either positive(Pos) or negative(Neg).

The program (perceplearn3.py) will perform document split to the word level token as features to train for classification learning and generate probabilities tables into file. 

The argument is a single file containing the training data; the program will learn vanilla and averaged models, and write the model parameters to a file called vanillamodel.txt and averagedmodel.txt.

This implementation support multiple categories of classification. For example, your data can be classifier as [['True', 'Fake'], ['Positive', 'Negative']]

## Smoothing:
The solution simply ignores unknown tokens in the test data.

## Tokenization: 
The program splits each word as a basic token unit. Certain punctuation will be removed, and lower casing all the letters. 
Stop word, high-frequency and low-frequency tokens will be ignored.


## Data:

A set of training and development data will be made available as separate files. 
1. The file (train-labeled.txt) is training data which include classification and reviews.
2. The file (dev-text.txt) is development set of data, with sequence no. and reviews which is separated by a new line. 
3. The file (dev-key.txt) is classifications for development set of data.
4. A readme/license file

## Programs:
There are two programs: perceplearn3.py will learn probabilities model from the training data, and the classifier program (percepclassify3.py) will read the model file and reviews to perform classifications. The learning program will be invoked in the following way: 
```
> python perceplearn3.py /path/to/input 
```
The argument is a single file containing the training data; the program will learn probabilities models, and write the model parameters to a file called vanillamodel.txt and averagedmodel.txt. The format of the model is JSON, and the model file should contain sufficient information for percepclassify3.py to successfully classify new data. 
The model file is human-readable, so that model parameters can be easily understood by visual inspection of the file. 


The classifier program will be invoked in the following way: 
```
> python percepclassify3.py /path/to/input [/path/to/answer]
```
The argument is a single file containing the test data; the program will read the parameters of a vanilla or averaged perceptron model from the file vanillamodel.txt or averagedmodel.txt, classify each review in the test data, and write the results to a text file called percepoutput.txt in the same format as the training data. 


## Usage: 
Training:
```
> python perceplearn3.py /path/to/input/training_corpus 
```

Classification:
```
> python percepclassify3.py /path/to/input/target_tokenized_file 
```

#### Input: 
1. text files 'train-labeled.txt' is a training corpus for English.
2. text files 'dev-text' is a test file which need to be classify for English.
3. text files 'dev-key.txt' is a answer/label file for test file.
4. model file 'vanillamodel.txt', 'averagedmodel.txt' will automatically generate and read by programs.

#### Output: Output a file with name percepoutput.txt in the same folder of the program.


