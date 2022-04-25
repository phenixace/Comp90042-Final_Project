from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import paddle

def calc_accuracy_score(true_labels, pred_labels):
    return accuracy_score(true_labels, pred_labels, normalize=False) / len(true_labels)

def calc_f1_score(true_labels, pred_labels):
    return f1_score(true_labels, pred_labels), precision_score(true_labels, pred_labels), recall_score(true_labels, pred_labels)

# this function is from baidu company <https://aistudio.baidu.com/aistudio/projectdetail/1968542>
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    losses = []
    result = []
    full_target = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        
        loss = criterion(logits, labels)
        losses.append(loss.numpy())

        pred = logits.argmax(1)

        result.extend(pred.cpu().tolist())
        full_target.extend(labels.cpu().tolist())

        correct = metric.compute(logits, labels)
        
        metric.update(correct)
        accu = metric.accumulate()

    precision = precision_score(full_target,result,average='macro')
    recall = recall_score(full_target,result,average='macro')

    f1 = f1_score(full_target,result,average='macro')
    print("eval loss: %.5f, accu: %.5f,  F1: %.4f, Precision: %.4f, Recall: %.4f" % (np.mean(losses), accu, f1, precision, recall))
    model.train()
    metric.reset()