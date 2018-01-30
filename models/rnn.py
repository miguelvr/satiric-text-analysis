import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.template import ModelTemplate
from models.custom_layers import Embedding
from data import SPECIAL_TOKENS
from data.features import \
    get_indexer, get_vocabulary, \
    load_polyglot, fit_vocabulary_to_embedding
from data.text_utils import recursive_map


class DocumentClassifier(ModelTemplate):
    def __init__(self, config=None, model_folder=None):
        super(DocumentClassifier, self).__init__(config, model_folder)
        self.vocabulary = None
        self.pretrained_embeddings = None
        self.loss_fn = None
        self.optimizer = None
        self.is_built = False

    def initialize_features(self, data=None, model_folder=None):
        if data is not None:
            self.vocabulary, _ = get_vocabulary(data['input'], flattened=True)
            if 'pretrained_embeddings' in self.config:
                polyglot_words, polyglot_vectors = \
                    load_polyglot(self.config['pretrained_embeddings'])
                self.pretrained_embeddings = fit_vocabulary_to_embedding(
                    self.vocabulary,
                    polyglot_words,
                    polyglot_vectors
                )
                self.pretrained_embeddings = torch.from_numpy(self.pretrained_embeddings).float()
            self.indexer = get_indexer(self.vocabulary)
        elif model_folder is not None:
            # FIXME
            raise NotImplementedError
        else:
            raise ValueError("Either data or model folder must be provided")

        self.initialized = True

    def build_model(self):
        pass

    def forward(self, *inputs):
        pass

    def update(self, input, output):
        self.train()

        input = Variable(input)
        tags = Variable(output)

        if self.gpu_device is not None:
            input = input.cuda()
            tags = tags.cuda()

        self.optimizer.zero_grad()  # Initialize Gradients to Zero
        out = self.forward(input)  # Forward Pass
        loss = self.loss_fn(out, tags)  # Compute Loss
        loss.backward()  # Backward Pass
        self.optimizer.step()  # optimizer Step

        return loss.data[0]

    def predict(self, input):
        self.eval()

        input = Variable(input)

        if self.gpu_device is not None:
            input = input.cuda()

        out = self.forward(input)  # Forward Pass
        scores = torch.exp(out).data.cpu().numpy()[:, 1].tolist()

        return scores

    def get_features(self, input=None, output=None):
        input = recursive_map(input, lambda x: self.indexer[x])
        max_length = max(map(len, input))
        padded_input = np.full((len(input), max_length), self.indexer[SPECIAL_TOKENS['padding']])
        for i, sent in enumerate(input):
            padded_input[i, :len(sent)] = np.array(sent)

        output = np.array(map(lambda x: 1. if x == 'satire' else 0., output))

        # Cast to torch tensors
        padded_input = torch.from_numpy(padded_input).long()
        output = torch.from_numpy(output).long()

        return {'input': padded_input, 'output': output}


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
        assert self.initialized, \
            "initialize_features() must be called before build_model()"

        # Layers
        if self.pretrained_embeddings is not None:
            self.embedding = Embedding(len(self.vocabulary), self.embedding_size,
                                       padding_idx=self.indexer[SPECIAL_TOKENS['padding']],
                                       pretrained=self.pretrained_embeddings)
        else:
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

        # Loss and optimizer
        balancing_weight = None
        if 'balancing_weight' in self.config:
            balancing_weight = torch.FloatTensor([1., self.config['balancing_weight']])

        self.loss_fn = nn.NLLLoss(weight=balancing_weight)
        self.optimizer = torch.optim.Adam(self.parameters())

        # Set model to a specific gpu device
        if self.gpu_device is not None:
            torch.cuda.set_device(self.gpu_device)
            self.cuda()

        self.is_built = True

    def forward(self, x):
        assert self.is_built, \
            "initialize_features() must be called before forward pass"

        # (batch, words) -> (batch, words, emb)
        h = self.embedding(x)

        # (batch, words, emb) -> (batch, words, hidden_size * num_directions)
        h, _ = self.rnn(h)

        # (batch, hidden_size * num_directions)
        h = h.sum(1, keepdim=False)

        # (batch, hidden_size * num_directions) -> (batch, 2)
        h = F.log_softmax(self.linear_out(h), dim=-1)

        return h


if __name__ == '__main__':
    x = Variable(torch.from_numpy(np.random.randint(10, size=(50, 10))).long())

    config = {
        'model_folder': 'tmp',
        'embedding_size': 64,
        'hidden_size': 20
    }

    rnn_classifier = RNNClassifier(config)
    rnn_classifier.build_model()

    print torch.exp(rnn_classifier(x))
