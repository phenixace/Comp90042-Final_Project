from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calc_accuracy_score(true_labels, pred_labels):
    return accuracy_score(true_labels, pred_labels)

def calc_f1_score(true_labels, pred_labels):
    return f1_score(true_labels, pred_labels), precision_score(true_labels, pred_labels), recall_score(true_labels, pred_labels)

