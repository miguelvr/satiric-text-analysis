from abc import ABCMeta, abstractmethod
import logging
import os
import yaml
import torch
import torch.nn as nn


class ModelTemplate(nn.Module):

    __metaclass__ = ABCMeta

    def __init__(self, config=None, model_folder=None):
        super(ModelTemplate, self).__init__()

        if config:
            self.config = config
            self.output_dir = config['model_folder']
        elif model_folder:
            config_dict = yaml.load(
                open('{}/config.yml'.format(model_folder), 'r')
            )
            self.config = config_dict
        else:
            raise ValueError(
                "Either a config or a model_folder have to "
                "be provided to initialize the model"
            )

        self.initialized = False

        self.gpu_device = None
        if 'gpu_device' in config:
            self.gpu_device = config['gpu_device']

        self.seed = 1234
        if 'seed' in config:
            self.seed = config['seed']

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    @abstractmethod
    def initialize_features(self, *inputs):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def update(self, input, output):
        pass

    @abstractmethod
    def predict(self, input):
        pass

    @abstractmethod
    def get_features(self):
        pass

    def load(self, path):
        logging.info("Loading Model Weights on {}".format(path))
        self.load_state_dict(
            torch.load(path, map_location=lambda storage, loc: storage)
        )

    def save(self):
        path = '%s/%s' % (self.output_dir, type(self).__name__.lower())
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            os.system('mkdir -p {}'.format(parent))
        torch.save(self.state_dict(), '{}.torch'.format(path))
