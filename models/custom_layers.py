import torch.nn as nn


class Embedding(nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False, pretrained=None):

        super(Embedding, self).__init__(num_embeddings, embedding_dim,
                                        padding_idx, max_norm,
                                        norm_type, scale_grad_by_freq,
                                        sparse)

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained_embedding):

        assert isinstance(pretrained_embedding, type(self.weight.data)), \
            "pretrained_embedding (%s) must have the same type as the Embedding weight (%s)" % \
            (type(pretrained_embedding), type(self.weight.data))

        num_embeddings = pretrained_embedding.size(0)
        embedding_dim = pretrained_embedding.size(1)

        assert num_embeddings == self.num_embeddings and\
               embedding_dim == self.embedding_dim, \
            "Loaded pretrained embedding has different dimensions from the initialized embeddings layer" \
            "(%d, %d) != (%d, %d)" % (num_embeddings, embedding_dim,
                                      self.num_embeddings, self.embedding_dim)

        self.weight = nn.Parameter(pretrained_embedding)
