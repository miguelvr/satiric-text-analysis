import sys
import argparse
from data.loading import load_bow_data
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from models.utils import print_results


def argument_parser():
    # ARGUMENT HANDLING

    parser = argparse.ArgumentParser(
        prog='Run baseline models'
    )

    parser.add_argument('--train-dir',
                        help='train directory path',
                        type=str,
                        required=True)

    parser.add_argument('--train-class',
                        help='train class file path',
                        type=str,
                        required=True)

    parser.add_argument('--test-dir',
                        help='test directory path',
                        type=str,
                        required=True)

    parser.add_argument('--test-class',
                        help='test class file path',
                        type=str,
                        required=True)

    parser.add_argument('--use-tfidf',
                        help='use tf-idf features',
                        default=False,
                        action='store_true')

    args = parser.parse_args(sys.argv[1:])

    return args


def logistic_regression(x, y, x_dev, x_test):

    print "Training Logistic Regression"
    log_reg = LogisticRegression()
    log_reg.fit(x, y)

    pred_dev = log_reg.predict(x_dev)
    pred_test = log_reg.predict(x_test)

    return pred_dev, pred_test


def svc(x, y,  x_dev, x_test, balanced=False):
    print "Training SVM"
    if balanced:
        class_weight = 'balanced'
    else:
        class_weight = None

    svm = SVC(class_weight=class_weight)
    svm.fit(x, y)

    pred_dev = svm.predict(x_dev)
    pred_test = svm.predict(x_test)

    return pred_dev, pred_test


def naive_bayes(x, y, x_dev, x_test):
    print "Training Naive Bayes"
    nb = GaussianNB()
    nb.fit(x, y)

    pred_dev = nb.predict(x_dev)
    pred_test = nb.predict(x_test)

    return pred_dev, pred_test


if __name__ == '__main__':

    args = argument_parser()

    X_train, y_train, X_dev, \
    y_dev, X_test, y_test = load_bow_data(
        args.train_dir,
        args.train_class,
        args.test_dir,
        args.test_class,
        lemmatization=True,
        use_tfidf=args.use_tfidf
    )

    # Naive Bayes
    nb_dev, nb_test = naive_bayes(X_train, y_train, X_dev, X_test)
    print_results(y_dev, nb_dev)
    print_results(y_test, nb_test)

    # SVM
    svm_dev, svm_test = svc(X_train, y_train, X_dev, X_test, balanced=True)
    print_results(y_dev, svm_dev)
    print_results(y_test, svm_test)

    # Logistic Regression
    lr_dev, lr_test = logistic_regression(X_train, y_train, X_dev, X_test)
    print_results(y_dev, lr_dev)
    print_results(y_test, lr_test)
