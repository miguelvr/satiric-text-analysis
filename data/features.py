import numpy as np
from collections import Counter, defaultdict
import text_utils
from __init__ import SPECIAL_TOKENS


def get_vocabulary(tokenized_documents):
    word_counter = Counter()
    for document in tokenized_documents:
        for sentence in document:
            word_counter.update(sentence)

    return sorted(list(set(word_counter.keys()) | set(SPECIAL_TOKENS.values()))), word_counter


def fit_vocabulary_to_embedding(vocabulary, embedding_words, embedding_vectors):
    embedding_indexer = {word: idx for idx, word in enumerate(embedding_words)}
    new_vocabulary = sorted(list(set(vocabulary) & set(embedding_words)))
    new_embedding = np.zeros((len(new_vocabulary), embedding_vectors.shape[1]))
    for i, word in enumerate(vocabulary):
        if word in embedding_words:
            new_embedding[i, :] = embedding_vectors[embedding_indexer[word], :]

    return new_vocabulary, new_embedding


def load_fasttext_embedding(fasttext_file):
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


if __name__ == '__main__':
    load_fasttext_embedding('satire/wiki-news-300d-1M.vec')
