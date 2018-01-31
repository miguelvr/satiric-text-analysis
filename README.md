# Satire Text Analysis

This repo aims to solve the problem of detection of satire in news documents.
Different approaches are provided, ranging from simple models to more complex and deeper models.
Coded with `sklearn` and `pytorch`. 

### Requirements

 - Python 2.7
 - [PyTorch v3.0](http://pytorch.org)
 
### Installation
 
  - (Optional) Create virtualenv or conda environment
  - Install [PyTorch](http://pytorch.org) (follow instructions on their website)
  - Install other dependencies
    - `pip install -r requirements.txt`
  - Run `python init_nltk.py` to dowload NLTK dependencies


### Run baseline models

Download the data and unzip it into the repository:

  - `bash download_data.sh`
  - or download it from [here](https://people.eng.unimelb.edu.au/tbaldwin/resources/satire/)

Baseline models script (may take a couple of minutes):

    python run_baselines.py \
      --train-dir satire/training \
      --train-class satire/training-class \
      --test-dir satire/test \
      --test-class satire/test-class

### Run RNN model

RNN model script (recommended use of GPU, otherwise it will take a few minutes):

    python run_rnn.py \
      --train-dir satire/training \
      --train-class satire/training-class \
      --test-dir satire/test \
      --test-class satire/test-class \
      --model-config configs/rnn.yml \  # model configuration file
      --cuda {gpu_id} \  # to run with gpu
      --model-folder {model_folder} # to override default model output directory
      
## Report

### Introduction

Satirical news are considered to be a form of entertainment. However, 
if interpreted literally, it is potentially deceptive and harmful. 
In the case the reader is not able to interpret the satirical cues of the document, 
there is not much of a difference between satirical news and fake news. 
Therefore, for applications such as automatic fact checking, 
detecting satire in a given document can be a very important task.

In this project, we start by tackling the problem in the simplest way possible, 
in order to have a baseline, to then start adding complexity to the problem.

Satire detection can be viewed as a simple natural language processing (NLP) 
classification task, and as such, it can be split into two distinct phases:
  1. Feature extraction
  2. Model selection

In the next sections, we'll explain how we process the data, in order to extract 
representative features, and the models used to achieve our goal.

### Feature Extraction

#### Cleaning the data

In order to extract relevant features from the documents, it is necessary to clean 
the data and represent it in a way our models can understand it. This will allow our 
models to learn meaningful features and not overfit on irrelevant noise.

We start by tokenizing the data, with the NLTK tokenizer, separating each individual 
element in a string into a list of tokens. Aditionally, we use a regular expression to detect every form 
of numeric characters such as integers or floats and map them into a unique `__numeric__` token. 
Our final preprocessing step is lemmatization, being optional, but it is especially helpful for simpler models.
Lemmatization consists in representing groups of words by the same token, 
in a way similarly to what we have done with the numeric tokens. 

Lemmatization example:

    cars -> car
    drove -> drive

Additional steps such as de-capitalizing every token and removing some special characters could have been added as well. 
But for now, we are keeping it simple, as our dataset isn't very noisy.

#### Extracting the features

##### Bag of Words

One of the simplest ways of representing entire text documents is a **Bag of Words (BoW)**. 
To build a BoW representation, we first build a vocabulary, consisting of every token in our training data. 
We will then do a one-hot encoding of our data, representing each token of the vocabulary as separate feature dimension. 
We'll then count the number of times each word appears in each document. Our BoW features will consist of a sparse matrix with shape 
`(#documents, #words_in_vocabulary)` which for each row will have the number of appearences of a token in its respective column and zeros elsewhere. This is called a Bag of Words, because it represent the documents in a way that completely ignores the order of words. This is illustrated below.

    document = ['This is a sentence!', 'Another sentence']
    vocabulary = ['!', 'a', 'Another', 'is', 'sentence', 'this']
    bow = [[1, 1, 0, 1, 1, 1],
           [0, 0, 1, 0, 1, 0]]

##### TF-IDF

The BoW representation treats every word equally, however, in most NLP applications, this is not a desirable behaviour. Some, more common, words simply do not convey a lot of meaning and are only introducing noise in our model. One simple way of helping our model focus on more meaningful words is to use the [**Term Frequency - Inverse Document Frequency (TF-IDF)**](https://en.wikipedia.org/wiki/Tf–idf) score. We use the TF-IDF scoring function on our BoW representation in order to weigh down the most common words across the documents and highlight the less frequent ones. 

Using TF-IDF might not be the most helpful weighting scheme for satire classification, as the satiric cues may not be in single tokens but instead in the way the sentences are constructed. Nonetheless, we decided to give this popular weighting scheme a try.

##### Word Embeddings

The previous representations will not be helpful when a model sees an unknown word, even if it has seen very similar words during training. To solve this problem, we need to capture the semantic meaning of words, meaning we need to understand that words like ‘good’ and ‘positive’ are closer than ‘peace’ and ‘war.’

**To be continued**

### Model Selection

### Results

### Conclusions
