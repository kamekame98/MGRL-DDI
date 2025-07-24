import numpy as np
from sklearn import metrics

def do_compute_metrics(probas_pred, target):

    pred = np.argmax(probas_pred, axis=1)

    num_classes = probas_pred.shape[1]
    target_onehot = np.eye(num_classes)[target]


    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target_onehot, probas_pred, average='macro', multi_class='ovr')
    f1 = metrics.f1_score(target, pred, average='macro')
    ap = metrics.average_precision_score(target_onehot, probas_pred, average='macro')
    
    return acc, auroc, f1,  ap

def positive(y_true):
    return np.sum((y_true == 1))

def negative(y_true):
    return np.sum((y_true == 0))

def true_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 1))

def false_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 1))

def true_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 0))

def false_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 0))

def accuracy(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s

    return np.sum((y_true == y_pred)) / sample_count

def sensitive(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    p = positive(y_true) + 1e-9
    return tp / p

def specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    n = negative(y_true) + 1e-9
    return tn / n

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = (2 * prec * reca) / (prec + reca)
    return fs
