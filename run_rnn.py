from __future__ import division
import sys
import argparse
import yaml
import numpy as np
from data.text_utils import flatten_list
from data.loading import load_split_data, get_batches
from models.rnn import RNNClassifier
from models.logger import ClassificationLogger
from models.trainer import Trainer
from models.utils import print_results


def argument_parser():
    # ARGUMENT HANDLING

    parser = argparse.ArgumentParser(
        prog='Train and test an RNN model for satiric text analysis'
    )

    parser.add_argument('--train-dir',
                        help='train directory path',
                        type=str,
                        required=True)

    parser.add_argument('--train-class',
                        help='train class file path',
                        type=str,
                        required=True)

    parser.add_argument('--test-dir',
                        help='test directory path',
                        type=str,
                        required=True)

    parser.add_argument('--test-class',
                        help='test class file path',
                        type=str,
                        required=True)

    parser.add_argument('--model-config',
                        help='model configuration path',
                        type=str,
                        required=True)

    parser.add_argument('--model-folder',
                        help='overloads model output folder config parameter',
                        type=str)

    parser.add_argument('--cuda',
                        help='overloads gpu_device config parameter',
                        type=int)

    args = parser.parse_args(sys.argv[1:])

    return args


if __name__ == '__main__':
    args = argument_parser()

    config = yaml.load(open(args.model_config, 'r'))

    if args.model_folder:
        config['model_folder'] = args.model_folder

    if args.cuda:
        config['gpu_device'] = args.cuda

    print "Loading Data"
    datasets, nr_samples = load_split_data(args.train_dir,
                                           args.train_class,
                                           args.test_dir,
                                           args.test_class)

    print "Building Model"
    model = RNNClassifier(config)
    model.initialize_features(data=datasets['train'])
    model.build_model()

    print "Extracting Features"
    train_data = get_batches(datasets['train'],
                             batch_size=config['batch_size'],
                             model_features=model.get_features)

    dev_data = get_batches(datasets['dev'],
                           batch_size=config['batch_size'],
                           model_features=model.get_features,
                           raw_output=True)

    test_data = get_batches(datasets['test'],
                            batch_size=config['batch_size'],
                            model_features=model.get_features,
                            raw_output=True)

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

    model.load('{}/{}.torch'.format(config['model_folder'], type(model).__name__.lower()))

    # Test
    predictions = flatten_list(trainer.test(test_data))
    pred_tags = np.where(np.array(predictions) >= 0.5, 1., 0.)
    gold_tags = np.array(flatten_list(map(lambda x: x['output'], test_data)))
    print_results(gold_tags, pred_tags)
