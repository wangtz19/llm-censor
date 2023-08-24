from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch
from tqdm import tqdm
import evaluate

checkpoint = "../results/23.08.24-14:43/checkpoint-17152"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")


test_data = json.load(open("../data/dev.json", "r", encoding="utf-8"))
labels, preds = [], []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
with torch.no_grad():
    for item in tqdm(test_data):
        text = item["text"]
        label = item["label"]
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True,
                           max_length=512, # 512 is the maximum length of BERT
                           padding="max_length").to(device)
        outputs = model(**inputs)
        logits = outputs.logits[0]
        pred = torch.argmax(logits).item()
        labels.append(label)
        preds.append(pred)


cls_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
print(cls_metrics.compute(predictions=preds, references=labels))
