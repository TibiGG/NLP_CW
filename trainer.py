# %%
from simpletransformers.classification import ClassificationModel, \
    ClassificationArgs
import pandas as pd
import logging
import torch
from collections import Counter
import os

# %%
# prepare logger
logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# check gpu
cuda_available = torch.cuda.is_available()

print('Cuda available? ', cuda_available)

# %%
if cuda_available:
    import tensorflow as tf

    # Get the GPU device name.
    device_name = tf.test.gpu_device_name()
    # The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')


# %% md
# Import Don't Patronize Me! data manager module

# %%
# helper function to save predictions to an output file
def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi]) + '\n')


# %%
from dont_patronize_me import DontPatronizeMe

# %%
dpm = DontPatronizeMe('.', '.')
# %%
dpm.load_task1()
dpm.load_task2(return_one_hot=True)

# %% md
# Load paragraph IDs
# %%
trids = pd.read_csv('train_semeval_parids-labels.csv')
teids = pd.read_csv('dev_semeval_parids-labels.csv')

# %%
trids.head()

# %%
trids.par_id = trids.par_id.astype(str)
teids.par_id = teids.par_id.astype(str)


# %% md
def rebuild_set(ids):
    rows = []  # will contain par_id, label and text
    for idx in range(len(ids)):
        parid = ids.par_id[idx]
        # print(parid)
        # select row from original dataset to retrieve `text` and binary label
        text = \
            dpm.train_task1_df.loc[
                dpm.train_task1_df.par_id == parid].text.values[
                0]
        label = \
            dpm.train_task1_df.loc[
                dpm.train_task1_df.par_id == parid].label.values[
                0]
        rows.append({
            'par_id': parid,
            'text': text,
            'label': label
        })
    return rows


# %%
# Rebuild training set (Task 1)
rows = rebuild_set(trids)
len(rows)
trdf1 = pd.DataFrame(rows)

# %%
# Rebuild test set (Task 1)
rows = rebuild_set(teids)
len(rows)
tedf1 = pd.DataFrame(rows)

# %% md
# RoBERTa Baseline for Task 1

# %%
# downsample negative instances
pcldf = trdf1[trdf1.label == 1]
nopcldf = trdf1[trdf1.label == 0]
npos = len(pcldf)

training_set1 = pd.concat([pcldf, nopcldf[:npos * 2]])

# %%
# Split the training data into train and validation sets
n_examples = training_set1.shape[0]
proportion_for_validation = 0 # between 0 and 1
n_validation = int(n_examples * proportion_for_validation)

if proportion_for_validation > 0:
    indices = np.random.permutation(n_examples)
    validation_subset = training_set1[['text', 'label']].iloc[indices[:n_validation]]
    training_subset = training_set1[['text', 'label']].iloc[indices[n_validation:]]

# %% md
# Training Code

# %%
task1_model_args = ClassificationArgs(num_train_epochs=1,
                                      no_save=True,
                                      no_cache=True,
                                      overwrite_output_dir=True)
task1_model = ClassificationModel("distilbert",
                                  'distilbert-base-uncased',
                                  args=task1_model_args,
                                  num_labels=2,
                                  use_cuda=cuda_available)

# train model
if proportion_for_validation == 0:
    task1_model.train_model(training_set1[['text', 'label']])
else:
    task1_model.train_model(train_df=training_subset, eval_df=validation_subset)

# run predictions
preds_task1, _ = task1_model.predict(tedf1.text.tolist())

# %%
print(Counter(preds_task1))

# Evaluate predictions
true_positive = ((preds_task1 == 1) & (tedf1.label == preds_task1)).sum() / (preds_task1 == 1).sum()
false_positive = ((preds_task1 == 1) & (tedf1.label != preds_task1)).sum() / (preds_task1 == 1).sum()
true_negative = ((preds_task1 == 0) & (tedf1.label == preds_task1)).sum() / (preds_task1 == 0).sum()
false_negative = ((preds_task1 == 0) & (tedf1.label != preds_task1)).sum() / (preds_task1 == 0).sum()
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + true_negative)
accuracy = (tedf1.label == preds_task1).mean()
f1_score = 2 * precision * recall / (precision + recall)
print("Proportion of correctly predicted labels:", accuracy)
print("F1-score:", f1_score)

# %%
labels2file([[k] for k in preds_task1], 'task1.txt')

# %%
# Prepare submission
os.system("cat task1.txt | head -n 10")
os.system("zip submission.zip task1.txt")
