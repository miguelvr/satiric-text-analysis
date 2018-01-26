import codecs
from spacy.en import English


parser = English()


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
        paragraph = []
        for line in f:
            if line == u'\n':
                document.append(" ".join(paragraph))
                paragraph = []
            else:
                paragraph.append(line.strip())

    return document


def tokenize_and_lemmatize(document, lemmatize=False):

    tokenized_document = []
    for paragraph in document:
        # get the tokens using spaCy
        parsed_doc = parser(paragraph)

        tokenized_paragraph = []
        for sent in parsed_doc.sents:
            if lemmatize:
                # lemmatize
                lemmas = []
                for token in sent:
                    lemmas.append(token.lemma_.lower().strip())
                tokenized_paragraph.append(lemmas)
            else:
                tokenized_paragraph.append(list(sent))

        tokenized_document.append(tokenized_paragraph)

    return tokenized_document


if __name__ == '__main__':
    documents, tags = read_dataset('satire/test', 'satire/test-class')
    print tokenize_and_lemmatize(documents[1])
