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

    return sorted(list(set(word_counter.keys()) | set(SPECIAL_TOKENS.values()))), word_counter


def bag_of_words(tokenized_documents, vocabulary, tfidf=False):

    indexer = get_indexer(vocabulary)
    documents_indices = text_utils.recursive_map(tokenized_documents, lambda x: indexer[x])

    vocabulary_length = len(vocabulary)

    documents_bow = []
    for doc in documents_indices:
        bow = np.zeros((len(doc), vocabulary_length))
        for i, sent_indices in enumerate(doc):
            bow[i, np.array(sent_indices, dtype=int)] += 1.0
        documents_bow.append(bow)

    if tfidf:
        idf = get_idf_mapper(tokenized_documents, vocabulary)
        for doc in documents_bow:
            for idx in range(doc.shape[1]):
                doc[:, idx] *= idf[vocabulary[idx]]

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

    from data.text_utils import load_tokenized_dataset, replace_numeric_tokens

    documents, tags = load_tokenized_dataset('satire/dbg', 'satire/dbg-class')
    documents = replace_numeric_tokens(documents)

    vocab, wc = get_vocabulary(documents)

    docs_bow = bag_of_words(documents, vocab)
    docs_bow = flatten_bow(docs_bow)

    print ""