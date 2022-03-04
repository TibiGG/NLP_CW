# %%
# Import dependencies
from multitask import MultiTaskModel
from back2back import Back2BackTranslator
from dont_patronize_me import DontPatronizeMe

import transformers
from transformers import \
    AutoTokenizer, \
    AutoConfig, \
    DataCollatorWithPadding, \
    AutoModelForSequenceClassification, \
    TrainingArguments, \
    Trainer
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

import numpy as np
import pandas as pd

import logging
from collections import Counter
import os
os.environ["WANDB_DISABLED"] = "true"

from pathlib import Path

from tqdm.notebook import tqdm
tqdm.pandas()


# %%
# prepare logger
logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
data_path = Path('./datasets/')

# check gpu
cuda_available = torch.cuda.is_available()

print('Cuda available? ', cuda_available)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Declare constants
model_name = "roberta-base"

epochs = 1
batch_size = 8
learning_rate = 2e-5
weight_decay = 0.01

# %%
# Load dataset
pcl_dataset = load_dataset('csv', data_files={'train': str(
    data_path/'dpm_pcl_train.csv'), 'test': str(data_path/'dpm_pcl_test.csv')})
pcl_emotion_dataset = load_dataset('csv', data_files=str(data_path/"dpm_pcl_emotion_train.csv"))

# %%
# Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess(example):
    return tokenizer(example['text'], truncation=True)


# %%
# Tokenize dataset
token_pcl_dataset = pcl_dataset.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# Instantiate the model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

num_training_steps = epochs * len(token_pcl_dataset['train'])
optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)
scheduler = transformers.get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# %%
# Define training args and trainer
training_args = TrainingArguments(
    report_to=None,
    output_dir='./results',
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=weight_decay,
    logging_steps=100,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=token_pcl_dataset['train'],
    eval_dataset=token_pcl_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # optimizers=(optimizer, scheduler),
)
trainer.train()

# %%
# Get predictions
predictions = trainer.predict(token_pcl_dataset["test"])
preds = np.argmax(predictions.predictions, axis=-1)

# %%
# Compute F1 score
score = f1_score(preds, predictions.label_ids, average='binary')
print(f"F1 score: {score}")

# %%
# Prepare predictions for submission
def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi]) + '\n')

labels2file([[k] for k in preds], "task1.txt")
os.system("cat task1.txt | head -n 10")
os.system("zip submission.zip task1.txt")
# %%
