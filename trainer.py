# %%
from dont_patronize_me import DontPatronizeMe
from simpletransformers.classification import ClassificationModel, \
    ClassificationArgs
import pandas as pd
import logging
import torch
from collections import Counter
import os
from pathlib import Path
from tqdm.notebook import tqdm
tqdm.pandas()

from back2back import Back2BackTranslator

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

# helper function to save predictions to an output file
def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi]) + '\n')


# %%
# Import Don't Patronize Me manager
dpm = DontPatronizeMe(data_path, data_path)

dpm.load_task1()
dpm.load_task2(return_one_hot=True)

# %%
# Load paragraph IDs

train_ids = pd.read_csv(data_path/'train_semeval_parids-labels.csv')
test_ids = pd.read_csv(data_path/'dev_semeval_parids-labels.csv')

train_ids.par_id = train_ids.par_id.astype(str)
test_ids.par_id = test_ids.par_id.astype(str)

# %%


def rebuild_set(ids):
    rows = []  # will contain par_id, label and text
    for idx in range(len(ids)):
        parid = ids.par_id[idx]
        # print(parid)
        # select row from original dataset to retrieve `text` and binary label
        text = \
            dpm.train_task1_df.loc[
                dpm.train_task1_df.par_id == parid].text.values[0]
        label = \
            dpm.train_task1_df.loc[
                dpm.train_task1_df.par_id == parid].label.values[0]
        rows.append({
            'text': text,
            'label': label
        })
    return rows


# %%
# Rebuild training set (Task 1)
rows = rebuild_set(train_ids)
print(len(rows))
train_data_df = pd.DataFrame(rows)

# %%
# Rebuild test set (Task 1)
rows = rebuild_set(test_ids)
print(len(rows))
test_data_df = pd.DataFrame(rows)

# %%
# Initialise translator with back2back capabilities
b2b = Back2BackTranslator()

# %%
# TODO: replace current logic with data augmentation
# downsample negative instances
train_pcl_df = train_data_df[train_data_df.label == 1]
train_nopcl_df = train_data_df[train_data_df.label == 0]

training_set1 = pd.concat([train_pcl_df, train_nopcl_df])

# %%
# Training Code

task1_model_args = ClassificationArgs(num_train_epochs=1,
                                      no_save=True,
                                      no_cache=True,
                                      overwrite_output_dir=True)
print(task1_model_args)
task1_model = ClassificationModel("distilbert",
                                  "distilbert-base-uncased",
                                  args=task1_model_args,
                                  num_labels=2,
                                  use_cuda=cuda_available)
# %%
# train model
task1_model.train_model(training_set1[['text', 'label']])
# run predictions
preds_task1, _ = task1_model.predict(test_data_df.text.tolist())

# %%
print(Counter(preds_task1))

# %%
# Evaluate predictions
true_positive = ((preds_task1 == 1) & (test_data_df.label == preds_task1)).sum() / (
    preds_task1 == 1).sum()
false_positive = ((preds_task1 == 1) & (test_data_df.label != preds_task1)).sum() / (
    preds_task1 == 1).sum()
true_negative = ((preds_task1 == 0) & (test_data_df.label == preds_task1)).sum() / (
    preds_task1 == 0).sum()
false_negative = ((preds_task1 == 0) & (test_data_df.label != preds_task1)).sum() / (
    preds_task1 == 0).sum()
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + true_negative)
accuracy = (test_data_df.label == preds_task1).mean()
f1_score = 2 * precision * recall / (precision + recall)
print("Proportion of correctly predicted labels:", accuracy)
print("F1-score:", f1_score)

# %%
labels2file([[k] for k in preds_task1], 'task1.txt')

# %%
# Prepare submission
os.system("cat task1.txt | head -n 10")
os.system("zip submission.zip task1.txt")
