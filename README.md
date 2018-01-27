## Satire Text Analysis

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
 - [Download data here](https://people.eng.unimelb.edu.au/tbaldwin/resources/satire/)

Example script:

    python run_baselines.py \
      --train-dir satire/training \
      --train-class satire/training-class \
      --test-dir satire/test \
      --test-class satire/test-class
