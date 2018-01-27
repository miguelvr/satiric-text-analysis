import numpy as np
from collections import Counter, defaultdict
import text_utils


SPECIAL_TOKENS = {
    'unknown': '__unknown__',
    'numeric': '__numeric__'
}


def get_vocabulary(tokenized_documents):
    word_counter = Counter()
    for document in tokenized_documents:
        for sentence in document:
            word_counter.update(sentence)

    return sorted(list(word_counter.keys())), word_counter


def one_hot_encoding(tokenized_documents, vocabulary):

    indexer = get_indexer(vocabulary)
    documents_indices = text_utils.recursive_map(tokenized_documents, lambda x: indexer[x])

    vocabulary_length = len(vocabulary)

    documents_one_hot = []
    for doc in documents_indices:
        one_hot = np.zeros((len(doc), vocabulary_length))
        for i, sent_indices in enumerate(doc):
            one_hot[i, np.array(sent_indices, dtype=int)] = 1.0
        documents_one_hot.append(one_hot)

    return documents_one_hot


def get_indexer(vocabulary):
    indexer = defaultdict(lambda: SPECIAL_TOKENS['unknown'])
    indexer.update({token: idx for idx, token in enumerate(vocabulary)})
    return indexer


if __name__ == '__main__':

    from data.text_utils import load_tokenized_dataset, replace_numeric_tokens

    documents, tags = load_tokenized_dataset('satire/dbg', 'satire/dbg-class')
    documents = replace_numeric_tokens(documents)

    vocab, wc = get_vocabulary(documents)

    print one_hot_encoding(documents, vocab)
