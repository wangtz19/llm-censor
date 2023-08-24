import evaluate
import numpy as np

id2label = {
    0: "NORMAL",
    1: "UNSAFE"
}

label2id = {
    "NORMAL": 0,
    "UNSAFE": 1
}

def set_proxy():
    import os
    proxy = 'http://dell-1.star:7890' # 3090 docker
    os.environ['http_proxy'] = proxy 
    os.environ['HTTP_PROXY'] = proxy
    os.environ['https_proxy'] = proxy
    os.environ['HTTPS_PROXY'] = proxy
set_proxy()

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)
