import numpy as np
import cPickle as pkl
from collections import Counter, defaultdict
import text_utils
from __init__ import SPECIAL_TOKENS


def get_vocabulary(tokenized_documents, flattened=False):
    word_counter = Counter()
    if flattened:
        for document in tokenized_documents:
            word_counter.update(document)
    else:
        for document in tokenized_documents:
            for sentence in document:
                word_counter.update(sentence)

    return sorted(list(set(word_counter.keys()) | set(SPECIAL_TOKENS.values()))), word_counter


def fit_vocabulary_to_embedding(vocabulary, embedding_words, embedding_vectors):
    embedding_indexer = {word: idx for idx, word in enumerate(embedding_words)}
    new_embedding = np.zeros((len(vocabulary), embedding_vectors.shape[1]))
    for i, word in enumerate(vocabulary):
        if word in embedding_words:
            new_embedding[i, :] = embedding_vectors[embedding_indexer[word], :]

    return new_embedding


def load_fasttext_embeddings(fasttext_file):
    fasttext_words = []
    with open(fasttext_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                fasttext_vocab_size, fasttext_dim = map(int, line.strip().split(' '))
                fasttext_embeddings = np.zeros((fasttext_vocab_size, fasttext_dim))
                continue
            splitLine = line.split(' ')
            word = splitLine[0]
            embedding = np.fromstring(" ".join(splitLine[1:]), dtype=float, sep=' ')
            fasttext_embeddings[i, :] = embedding
            fasttext_words.append(word)
    return fasttext_words, fasttext_embeddings


def load_polyglot(polyglot_format_file):
    """
    Loads the polyglot embeddings vectors and vocabulary,
    mapping special tokens to our nomenclature

    Args:
        polyglot_format_file:   (string) file path

    Returns:
        polyglot_words:         list of words
        polyglot_vectors:       numpy array

    """

    polyglot_words, polyglot_vectors = pkl.load(open(polyglot_format_file, 'r'))

    polyglot_map = {
        u'<UNK>': u'unknown',
        u'<S>': u'sentence-start',
        u'</S>': u'sentence-end',
        u'<PAD>': u'padding'
    }

    polyglot_words = [SPECIAL_TOKENS[polyglot_map[word]]
                      if word in polyglot_map else word for word in polyglot_words]

    return polyglot_words, polyglot_vectors


def bag_of_words(tokenized_documents, vocabulary):
    indexer = get_indexer(vocabulary)
    documents_indices = text_utils.recursive_map(tokenized_documents, lambda x: indexer[x])

    vocabulary_length = len(vocabulary)

    documents_bow = []
    for doc in documents_indices:
        bow = np.zeros((len(doc), vocabulary_length))
        for i, sent_indices in enumerate(doc):
            bow[i, np.array(sent_indices, dtype=int)] += 1.0
        documents_bow.append(bow)

    return documents_bow


def flatten_bow(documents_bow):
    flattened_bow = []
    for doc in documents_bow:
        flattened_bow.append(np.sum(doc, axis=0))

    return np.stack(flattened_bow)


def get_indexer(vocabulary):
    indexer = defaultdict(lambda: vocabulary.index(SPECIAL_TOKENS['unknown']))
    indexer.update({token: idx for idx, token in enumerate(vocabulary)})
    return indexer


def tfidf(flattened_bow):
    """Term Frequency - Inverse Document Frequency"""
    tf = flattened_bow
    n_docs = flattened_bow.shape[0]
    doc_counts = np.count_nonzero(flattened_bow, axis=0)
    doc_counts[np.where(doc_counts == 0)[0]] = 1
    idf = 1. + (np.log(n_docs / doc_counts))
    return tf * idf

