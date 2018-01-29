from __future__ import division
import math
from data.text_utils import \
    load_tokenized_dataset, replace_numeric_tokens, \
    lemmatize, flatten_documents
from models.rnn import RNNClassifier
from models.logger import ClassificationLogger
from models.trainer import Trainer


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


def get_batches(dset_dict, batch_size=1, model_features=None, no_output=False):
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

        data.append(feat_batch)

    return data


if __name__ == '__main__':
    print "Loading Data"
    datasets, nr_samples = load_split_data('satire/dbg',
                                           'satire/dbg-class',
                                           'satire/dbg',
                                           'satire/dbg-class')

    print "Building Model"
    config = {
        'model_folder': 'tmp',
        'embedding_size': 64,
        'hidden_size': 20,
        'batch_size': 2,
        'epochs': 10,

    }

    model = RNNClassifier(config)
    model.initialize_features(data=datasets['train'])
    model.build_model()

    print "Extracting Features"
    train_data = get_batches(datasets['train'],
                             batch_size=config['batch_size'],
                             model_features=model.get_features)

    dev_data = get_batches(datasets['dev'],
                           batch_size=config['batch_size'],
                           model_features=model.get_features)

    test_data = get_batches(datasets['test'],
                            batch_size=config['batch_size'],
                            model_features=model.get_features)

    print "Training Model"
    logger = ClassificationLogger(
        monitoring_metric='f1_product',
        nr_samples=nr_samples['train'],
        batch_size=config['batch_size']
    )

    trainer = Trainer(model, logger)
    trainer.fit(train_data=train_data,
                dev_data=dev_data,
                epochs=config['epochs'])
