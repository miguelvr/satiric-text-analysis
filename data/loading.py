import math
import numpy as np
from data.features import \
    get_vocabulary, bag_of_words, flatten_bow, tfidf
from data.text_utils import \
    load_tokenized_dataset, replace_numeric_tokens, \
    lemmatize, flatten_documents


def load_bow_data(dir_train, file_class_train, dir_test,
                  file_class_test, lemmatization=False, use_tfidf=False):
    # Load
    print "Loading Data"
    documents, tags = load_tokenized_dataset(dir_train, file_class_train)
    documents = replace_numeric_tokens(documents)

    documents_test, tags_test = load_tokenized_dataset(dir_test, file_class_test)
    documents_test = replace_numeric_tokens(documents_test)

    if lemmatization:
        documents = lemmatize(documents)
        documents_test = lemmatize(documents_test)

    print "Building Vocabulary"
    vocabulary, word_counter = get_vocabulary(documents)

    print "Extracting Features"
    # Train
    # (#documents, #sentences, vocab_len)
    docs_bow = bag_of_words(documents, vocabulary)
    # (#documents, vocab_len)
    X = flatten_bow(docs_bow)
    y = np.array(map(lambda x: 1. if x == 'satire' else 0., tags))

    # Test
    docs_bow_test = bag_of_words(documents_test, vocabulary)
    X_test = flatten_bow(docs_bow_test)
    y_test = np.array(map(lambda x: 1. if x == 'satire' else 0., tags_test))

    if use_tfidf:
        X = tfidf(X)
        X_test = tfidf(X_test)

    return X, y, X_test, y_test


def load_split_data(train_dir, train_class, test_dir, test_class, lemmatization=False):
    train_data, train_tags = load_tokenized_dataset(train_dir, train_class)
    train_data = replace_numeric_tokens(train_data)
    train_data = flatten_documents(train_data)

    dev_test_data, dev_test_tags = load_tokenized_dataset(test_dir, test_class)
    dev_test_data = replace_numeric_tokens(dev_test_data)
    dev_test_data = flatten_documents(dev_test_data)

    if lemmatization:
        train_data = lemmatize(train_data)
        dev_test_data = lemmatize(dev_test_data)

    train = {'input': train_data, 'output': train_tags}

    dev_data = dev_test_data[:int(math.ceil(len(dev_test_data) / 2))]
    dev_tags = dev_test_tags[:int(math.ceil(len(dev_test_tags) / 2))]
    dev = {'input': dev_data, 'output': dev_tags}

    test_data = dev_test_data[int(math.ceil(len(dev_test_data) / 2)):]
    test_tags = dev_test_tags[int(math.ceil(len(dev_test_tags) / 2)):]
    test = {'input': test_data, 'output': test_tags}

    datasets = {
        'train': train,
        'dev': dev,
        'test': test
    }

    nr_samples = {
        'train': len(train_data),
        'dev': len(dev_data),
        'test': len(test_data)
    }

    return datasets, nr_samples


def get_batches(dset_dict, batch_size=1, model_features=None, no_output=False, raw_output=False):
    nr_examples = len(dset_dict['input'])
    if batch_size is None:
        nr_batch = 1
        batch_size = nr_examples
    else:
        nr_batch = int(math.ceil(nr_examples * 1. / batch_size))

    data = []
    # Ignore output when solicited
    if no_output:
        data_sides = ['input']
    else:
        data_sides = ['input', 'output']

    for batch_n in range(nr_batch):

        # Colect data for this batch
        data_batch = {}
        for side in data_sides:
            data_batch[side] = dset_dict[side][batch_n * batch_size:(batch_n + 1) * batch_size]

        # If feature extractors provided, return features instead
        if model_features is not None:
            feat_batch = model_features(**data_batch)
        else:
            feat_batch = data_batch

        if not no_output:
            if raw_output:
                # keep output unextracted
                feat_batch['output'] = map(lambda x: 1. if x == 'satire' else 0., data_batch['output'])

        data.append(feat_batch)

    return data
