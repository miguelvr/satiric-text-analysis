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

### Model Selection

### Results

### Conclusions
