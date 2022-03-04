# %%
# Import dependencies
from pyexpat import features
from multitask import MultiTaskModel
from back2back import Back2BackTranslator
from dont_patronize_me import DontPatronizeMe
from data_collator import MultitaskTrainer, PCLDataCollator

import transformers
from transformers import \
    AutoTokenizer, \
    AutoConfig, \
    DataCollatorWithPadding, \
    AutoModelForSequenceClassification, \
    AutoModelForMultipleChoice, \
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

dataset_dict = {
    "pcl_binary": pcl_dataset,
    "pcl_emotion": pcl_emotion_dataset,
}

# %%
# Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_pcl_binary(example):
    return tokenizer(example['text'], truncation=True)

def preprocess_pcl_emotion(example):
    inputs = example["text"]
    features = tokenizer(inputs, truncation=True)
    return features

convert_func_dict = {
    "pcl_binary": preprocess_pcl_binary,
    "pcl_emotion": preprocess_pcl_emotion,
}

# %%
# Tokenize dataset
columns_dict = {
    "pcl_binary": ['input_ids', 'attention_mask', 'label'],
    "pcl_emotion": ['input_ids', 'attention_mask', 'label'],
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False,
        )

# %%
# Instantiate the model
multitask_model = MultiTaskModel.create(
    model_name=model_name,
    model_type_dict={
        "pcl_binary": AutoModelForSequenceClassification,
        "pcl_emotion": AutoModelForMultipleChoice
    },
    model_config_dict={
        "pcl_binary": AutoConfig.from_pretrained(model_name, num_labels=2),
        "pcl_emotion": AutoConfig.from_pretrained(model_name, num_labels=7),
    }
)


# %%
# Define training args and trainer

train_dataset = {
    "pcl_binary": features_dict["pcl_binary"]["train"],
    "pcl_emotion": features_dict["pcl_emotion"]
}
trainer = MultitaskTrainer(
    model=multitask_model,
    args=TrainingArguments(
        output_dir="./multitask_results",
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
    ),
    data_collator=PCLDataCollator(),
    train_dataset=train_dataset,
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
# os.system("cat task1.txt | head -n 10")
os.system("zip submission.zip task1.txt")
# %%
