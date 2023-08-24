from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import id2label, label2id, compute_metrics, set_proxy
import datetime
from dataset import get_dataset, preprocess_dataset, split_dataset, get_collator
from logger import get_logger


logger = get_logger(__name__)
# set_proxy()

BERT_MODEL = "hfl/chinese-bert-wwm"
# BERT_MODEL = "distilbert-base-uncased"

logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

logger.info("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL, 
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

date_str = datetime.datetime.now().strftime("%y.%m.%d-%H:%M")
training_args = TrainingArguments(
    output_dir=f"../results/{date_str}",
    learning_rate=2e-5,
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir=f"../logs/{date_str}",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

logger.info("Loading dataset...")
dataset = get_dataset("../data/train.json")["train"]
dataset = preprocess_dataset(dataset, tokenizer)
dataset = split_dataset(dataset, 0.1, True)

logger.info("Getting collator...")
collator = get_collator(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train_dataset"],
    eval_dataset=dataset["eval_dataset"],
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer.train()
