import codecs
import nltk
import re
from __init__ import SPECIAL_TOKENS


def load_tokenized_dataset(data_dir, tags_file):
    documents, tags = read_dataset(data_dir, tags_file)

    tokenized_documents = recursive_map(documents, nltk.word_tokenize)

    return tokenized_documents, tags


def flatten_documents(documents):
    """Flattens the document structure
    (documents, sentences, tokens) -> (documents, sentences * tokens)"""
    flattened = []
    for document in documents:
        flattened.append([token for sentence in document for token in sentence])
    return flattened


def replace_numeric_tokens(documents):
    def replace_numeric(token, replace=SPECIAL_TOKENS['numeric']):
        if re.match(r"[-+]?\d*\.\d+|\d+|[-+]?\d*,\d+", token):
            return replace
        else:
            return token

    return recursive_map(documents, replace_numeric)


def lemmatize(documents):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return recursive_map(documents, lemmatizer.lemmatize)


def read_dataset(data_dir, tags_file):

    valid_file_names, tags = read_tags(tags_file)

    documents = []
    for name in valid_file_names:
        documents.append(read_document('%s/%s' % (data_dir, name)))

    return documents, tags


def read_tags(tags_file):
    valid_file_names = []
    tags = []
    with codecs.open(tags_file, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            name, tag = line.strip().split(' ')
            valid_file_names.append(name)
            tags.append(tag)

    return valid_file_names, tags


def read_document(document_file):
    document = []
    with codecs.open(document_file, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            document.append(line.strip())

    document = filter(lambda x: x != u'', document)

    return document


def recursive_map(l, f, dtype=list):
    if isinstance(l, dtype):
        return map(lambda x: recursive_map(x, f), l)
    return f(l)


if __name__ == '__main__':
    documents, tags = load_tokenized_dataset('satire/dbg', 'satire/dbg-class')
    documents = replace_numeric_tokens(documents)
    print documents[:3]
