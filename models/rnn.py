import torch
import torch.nn as nn
import torch.nn.functional as F
from models.template import ModelTemplate
from models.custom_layers import Embedding
from data import SPECIAL_TOKENS
from data.features import get_indexer, get_vocabulary


class DocumentClassifier(ModelTemplate):
    def __init__(self, config=None, model_folder=None):
        super(DocumentClassifier, self).__init__(config, model_folder)
        self.is_built = False

        self.vocabulary = None

    def initialize_features(self, data=None, model_folder=None):
        if data is not None:
            # FIXME: Choose function accordingly with pretrained embs
            self.vocabulary = get_vocabulary(data)
            self.indexer = get_indexer(self.vocabulary)
        elif model_folder is not None:
            raise NotImplementedError
        else:
            raise ValueError("Either data or model folder must be provided")

        self.build_model()

    def build_model(self):
        pass

    def forward(self, *inputs):
        pass

    def update(self, input, output):
        pass

    def predict(self, input, output):
        pass

    def get_features(self):
        pass


class RNNClassifier(DocumentClassifier):

    def __init__(self, config=None, model_folder=None):
        super(RNNClassifier, self).__init__(config, model_folder)

        self.embedding_size = self.config['embedding_size']
        self.hidden_size = self.config['hidden_size']

        self.num_layers = 1
        if 'num_layers' in self.config:
            self.num_layers = self.config['num_layers']

        self.bidirectional = False
        if 'bidirectional' in self.config:
            self.bidirectional = self.config['bidirectional']

        self.dropout = 0.
        if 'dropout' in self.config:
            self.dropout = self.config['dropout']

        self.embedding = None
        self.rnn = None
        self.linear_out = None

    def build_model(self):

        # TODO: load pretrained embedding

        self.embedding = Embedding(len(self.vocabulary), self.embedding_size,
                                   padding_idx=self.indexer[SPECIAL_TOKENS['padding']])
        self.rnn = nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True
        )
        if self.bidirectional:
            self.linear_out = nn.Linear(2 * self.hidden_size, 2)
        else:
            self.linear_out = nn.Linear(self.hidden_size, 2)

        self.is_built = True

    def forward(self, x):
        assert self.is_built, "build_model() must be called before forward pass"

        # (sents, words) -> (sents, words, emb)
        h = self.embedding(x)

        # (sent, words, emb) -> (sent, words, hidden_size * num_directions)
        h, _ = self.rnn(h)

        # (1, hidden_size * num_directions)
        h = h.sum(1, keepdim=False).sum(0)

        # (1, hidden_size * num_directions) -> (1, 2)
        h = F.log_softmax(self.linear_out(h), dim=-1)

        return h


if __name__ == '__main__':

    import numpy as np
    from torch.autograd import Variable

    x = Variable(torch.from_numpy(np.random.randint(10, size=(50, 10))).long())

    config = {
        'model_folder': 'tmp',
        'embedding_size': 64,
        'hidden_size': 20
    }

    rnn_classifier = RNNClassifier(config)
    rnn_classifier.build_model()

    print torch.exp(rnn_classifier(x))
