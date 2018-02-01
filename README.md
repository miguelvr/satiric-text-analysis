# Satiric Text Analysis

This repo aims to solve the problem of detection of satire in news documents.
Different approaches are provided, ranging from simple models to more complex and deeper models.
Coded with `sklearn` and `pytorch`. 

### Requirements

 - Python 2.7
 - [PyTorch v0.3](http://pytorch.org)
 
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
there is not much of a difference between satirical news and fake news, 
but it is a lot harder to compare the facts against a ground truth in 
satire detection, as there is not necessarily a ground truth.
Therefore, for applications such as automatic fact checking, 
detecting satire in a given document can be a very important task.

In this project, the problem is tackled in the simplest way possible, 
in order to have a baseline, and then start adding complexity.

Satire detection can be viewed as a simple natural language processing (NLP) 
classification task, and as such, it can be split into two distinct phases:
  1. Feature extraction
  2. Model selection

The next sections cover how the data is processed, in order to extract 
representative features, and the models chosen to achieve our goal.

### Feature Extraction

#### Cleaning the data

In order to extract relevant features from the documents, it is necessary to clean 
the data and represent it in a way our models can understand it. This will allow the 
models to learn meaningful features and not overfit on irrelevant noise.

Processing the data starts by tokenizing it, with the NLTK tokenizer, to separate each individual 
element in a string into a list of tokens. Aditionally, a regular expression is used to detect every form 
of numeric characters such as integers or floats and map them into a unique `__numeric__` token. 
The final preprocessing step is lemmatization, being optional, but is especially helpful 
for simpler models in many NLP tasks. Lemmatization consists in representing groups of words by the same token, 
in a way similarly to what is done with the numeric tokens. 
To deal with unknown tokens during test time, a special `__unk__` token is added.

Lemmatization example:

    cars -> car
    drove -> drive

Additional steps such as de-capitalizing every token and removing some special characters could have been added as well. 
However, as our dataset isn't very noisy, let's keep it simple.

#### Extracting the features

##### Bag of Words

One of the simplest ways of representing entire text documents is a **Bag of Words (BoW)**. 
To build a BoW representation, a vocabulary consisting of every token in our training data is created. 
Then a one-hot encoding is applied to the data, representing each token of the vocabulary as separate feature dimensions. 
Finally, a count of the number of times each word appears in each document is tracked. The BoW features consist of a sparse matrix with shape `(#documents, #words_in_vocabulary)` which for each row will have the number of appearences of a token in its respective column and zeros elsewhere. This is called a Bag of Words, because it represents the documents in a way that completely ignores the order of words. This is illustrated below.

    document = ['This is a sentence!', 'Another sentence']
    vocabulary = ['!', 'a', 'Another', 'is', 'sentence', 'this']
    bow = [[1, 1, 0, 1, 1, 1],
           [0, 0, 1, 0, 1, 0]]

##### TF-IDF

The BoW representation treats every word equally, however, in most NLP applications, this is not a desirable behaviour. Some, more common, words simply do not convey a lot of meaning and are only introducing noise in our model. One simple way of helping our model focus on more meaningful words is to use the [**Term Frequency - Inverse Document Frequency (TF-IDF)**](https://en.wikipedia.org/wiki/Tf–idf) score. The TF-IDF scoring function is used on the BoW representation in order to weigh down the most common words across the documents and highlight the less frequent ones.

Using TF-IDF might not be the most helpful weighting scheme for satire classification, as the satiric cues may not be in single tokens but instead in the way the sentences are constructed. Nonetheless, I decided to give this popular weighting scheme a try.

##### Word Embeddings (Continuous Bag of Words)

The previous representations will not be helpful when a model sees an unknown word, even if it has seen very similar words during training. To solve this problem, one needs to capture the semantic meaning of words, meaning we need to understand that words like ‘good’ and ‘positive’ are closer than ‘peace’ and ‘war.’ With word embeddinds it is possible to go from a discrete space with one dimension per word to a continuous vector space with much lower dimension, where words similar words are closer to each other in the embedding space. Using word embeddings, the documents can be represent as a **continuous bag of words (CBoW)**, in which each document is the sum of muiti-dimensional vectors in the embedding space.

There are several pretrained word embeddings with several million tokens, such as Word2Vec, GLoVe or FastText, which can be used with any model. For the purpose of this project, pretrained polyglot embeddings were used, because of their lower dimensionality relatively good performance.

##### Other relevant features

There are many other features that could be useful in a satire detection task, such as **N-grams** that would convey a temporal structure for simpler models, by grouping N tokens at a time, or linguistic features such as **part of speech tags**, that would help identifying writing styles related to satire and better define the role of each word in a sentence.

### Model Selection

#### Baseline Models

To start with, the extracted features were plugged in some, very simple, classical machine learning models for classification tasks: Naive Bayes, Support Vector Machines (SVM) and Logistic Regression. Often, when small amounts of data are provided, simple models perform better than more complex models. 

In a nutshell, Naive Bayes does counts over the data to estimate its priors and posterior probabilities to compute the likehood of a class given the data. SVM estimates a hyperplane that separates the different classes of data, in a kernel mapped hyperspace, by maximizing the margin of separation between points of different classes. Logistic regression binary classification can be interpreted as a non-linear regression problem, in which we map the data to approximate one of either of the classes `[0, 1]`. The closest the output of the logistic regression is to one of the boundary values imposed by the logistic function, the more confident that prediction is.

For each one of the baseline models, the correspondant implementations in `scikit-learn` were used (`MultinomialNB`, `SVC` and `LogisticRegression`).

#### Recurrent Neural Networks (RNN)

Since a document is a sequence of paragraphs, which is a sequence of sentences, which is a sequence of tokens, using a model that takes into account temporal sequence makes a lot of sense. RNNs are models that are able to capture long term dependencies in data and therefore are a great fit for the task of satire detection.

The implemented model consists of an embedding layer followed by multiple layers of unidirectional or bidirectional Long Short-Term Memory (LSTM) networks and ending in a linear layer with a Softmax activation. The model discards the paragraph and sentence structures of a document and deals with it as a sequence of tokens. The Embeddings size is 64 (same as polyglot) and the LSTM hidden 

The model is trained for about 30 epochs with GPU accelaration using an ADAM optimizer and a Negative Log Likelihood loss function with class weighting for the `satire` class.

### Results

The data is split into three partitions: train, dev and test. The dev and test partitions are then used to test the model, and for the case of the LSTM models, the dev partition is also used to track the best model.

Since the used dataset is very unbalanced, accuracy is not a good measure to keep track of the model's performance. Therefore the product of the F1 scores of both classes is used. The F1 product is a much more representative metric, as it translates to making correct predictions in both classes (`true` and `satire`).

The features used for the baseline models were the **BoW** features, as opposed to the **word embeddings** used to train the LSTM based models.

The results of the experiments conducted are presented below:

| Model                                       	| Dev Accuracy 	| Dev F1 Product 	| Test Accuracy 	| Test F1 Product 	|
|---------------------------------------------	|--------------	|----------------	|---------------	|-----------------	|
| Naive Bayes	                                 | 0.964	        | 0.687	          | 0.960	         | 0.457            |
| Naive Bayes + lemmatization	                 | 0.960	        | 0.687	          |	0.956	         |	0.434	           |
| Naive Bayes + lemmatization + tfidf	         | 0.970	        |	0.761	          |	0.967	         |	0.679            |
| SVM                                         	| 0.940        	| 0.640          	| 0.920         	| 0.455           	|
| SVM + lemmatization                         	| 0.941        	| 0.650          	| 0.919         	| 0.467           	|
| SVM + lemmatization + tfidf                 	| 0.970        	| 0.752          	| 0.965         	| 0.565           	|
| Logistic Regression                         	| **0.974**    	| 0.785          	| **0.977**     	| **0.734**       	|
| Logistic Regression + lemmatization         	| **0.974**    	| 0.785          	| 0.975         	| 0.705           	|
| Logistic Regression + lemmatization + tfidf 	| 0.962        	| 0.660          	| 0.966         	| 0.533           	|
| LSTM                                        	| 0.959        	| 0.735          	| 0.947         	| 0.626           	|
| LSTM + pre-trained embeddings               	| 0.971        	| 0.780          	| 0.970          | 0.656           	|
| BiLSTM                                      	| 0.970        	| **0.787**      	| 0.952         	| 0.589           	|
| BiLSTM + pre-trained embeddings             	| 0.970        	| 0.780          	| 0.961         	| 0.622           	|

From the results obtained, it is concluded that the simpler models, overall, perform better than the more complex LSTM model. This can be explained by the low amounts of data in our training set, as deep learning approaches are very data hungry. Despite this fact, the LSTM based model reaches comparable performances to the best model, the Logistic Regression, and it is the best performing model in the dev set. However, it is clear that the LSTMs are overfitting, as there's a considerable gap between the performance of dev and test. Further hyperparameter search could have been conducted to achieve better results, but considering the scope of this project, only a limited search was performed.

Having a simple model such as Logistic Regression to tackle a problem has a lot of advantages over, more sophisticated, deep learning approaches. It is much faster to train, corresponding to a small fraction of an LSTM training period. It is also possible to visualize the features that affect the model the most, as the model's parameters are a one dimensional vector, opposed to a deep learning model which is more of a "black box" due the large number of high dimensional parameters. This way, a logistic regression model is much more "explainable" than an LSTM model.

The top 10 most relevant features of our best performing model were analyzed and are presented below:

| Token 	| Weight 	|
|-------	|--------	|
| !     	| 0.629  	|
| '     	| 0.541  	|
| Bush  	| 0.514  	|
| $     	| 0.448  	|
| *     	| 0.430  	|
| Mr    	| 0.428  	|
| we    	| 0.425  	|
| my    	| 0.402  	|
| you   	| 0.394  	|
| in    	| 0.382  	|

This can be interpreted as satire detection being correlated with the occurance of exclamations, quotes, references to first and second person singular and articles refering to Bush. The latter, is tied to the time period of the corpus extraction.

### Conclusions and Future Work

Different models were trained with the scope of satire identification in text documents, ranging from classical approaches to deep learning techniques. The problem was tackled as a simple NLP classification task, and common feature extraction techniques were used and lead to a relatively performant model in the provided dataset, with 97.7% accuracy and 0.734 F1 product score obtained in the test set.

Although the model chosen performs well in the used data set, it is not expected that this model perform as well in real world data, as satire detection is tied to several other sub-problems such as irony detection and tone identification, which our model wont be able to deal with, especially if these are tied to unknown tokens. 

To improve results, it would be interesting to add tools such as slang identification and linguistic features such as POS tags to identify wrinting styles tied to satire. In addition, the deep learning could take advantage of the structure of the data and process paragraphs and sentences separately. It would be interesting to use an hierachical attention layer to make the model focus on specific zones of the document, which have satirical cues, as not all paragraphs are satire.
