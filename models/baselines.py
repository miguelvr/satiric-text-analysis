import numpy as np
from data.text_utils import load_tokenized_dataset, replace_numeric_tokens
from data.features import bag_of_words, get_vocabulary
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score


def load_data(dir_train, file_class_train, dir_test, file_class_test, flatten_operation='sum'):

    # Load
    print "Loading Data"
    documents, tags = load_tokenized_dataset(dir_train, file_class_train)
    documents = replace_numeric_tokens(documents)

    documents_test, tags_test = load_tokenized_dataset(dir_test, file_class_test)
    documents_test = replace_numeric_tokens(documents_test)

    print "Building Vocabulary"
    vocabulary, word_counter = get_vocabulary(documents)

    print "Extracting Features"
    # Train
    # (#documents, #sentences, vocab_len)
    docs_bow = bag_of_words(documents, vocabulary)
    # (#documents, vocab_len)
    X = flatten_bow(docs_bow, op=flatten_operation)
    y = np.array(map(lambda x: 1. if x == 'satire' else 0., tags))

    # Test
    docs_bow_test = bag_of_words(documents_test, vocabulary)
    X_test = flatten_bow(docs_bow_test, op=flatten_operation)
    y_test = np.array(map(lambda x: 1. if x == 'satire' else 0., tags_test))

    return X, y, X_test, y_test


def flatten_bow(documents_bow, op='sum'):

    if op == 'sum':
        f = np.sum
    elif op == 'mean':
        f = np.mean
    else:
        raise ValueError("Invalid op: %s" % op)

    flattened_bow = []
    for doc in documents_bow:
        flattened_bow.append(f(doc, axis=0))

    return np.stack(flattened_bow)


def print_results(y_true, y_pred):

    f1 = f1_score(y_true, y_pred, average=None)
    f1_product = f1[0] * f1[1]

    print "Result:"
    print classification_report(y_true, y_pred, target_names=['true', 'satire'])
    print "Accuracy: %.3f" % accuracy_score(y_true, y_pred)
    print "F1 Procuct: %.3f " % f1_product


def logistic_regression(x, y, x_test):

    print "Training Logistic Regression"
    log_reg = LogisticRegression()
    log_reg.fit(x, y)

    prediction = log_reg.predict(x_test)

    return prediction


if __name__ == '__main__':

    X_train, y_train, X_test, y_test = load_data(
        'satire/training',
        'satire/training-class',
        'satire/test',
        'satire/test-class',
    )

    log_reg_prediction = logistic_regression(X_train, y_train, X_test)

    print_results(y_test, log_reg_prediction)
