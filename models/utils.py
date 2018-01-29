from sklearn.metrics import classification_report, accuracy_score, f1_score


def print_results(y_true, y_pred):

    f1 = f1_score(y_true, y_pred, average=None)
    f1_product = f1[0] * f1[1]

    print "Result:"
    print classification_report(y_true, y_pred, target_names=['true', 'satire'])
    print "Accuracy: %.3f" % accuracy_score(y_true, y_pred)
    print "F1 Procuct: %.3f " % f1_product
