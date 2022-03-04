#!/usr/bin/env python3
# %%
import pandas as pd

# %%
df_pcl = pd.read_csv('./datasets/dpm_pcl.csv', index_col=False)
df_pcl['text'] = df_pcl['text'].astype(str)
df_pcl['label'] = df_pcl['label'].apply(lambda x: 1 if x else 0)

# %%
train_ids = pd.read_csv('./datasets/train_semeval_parids-labels.csv')
test_ids = pd.read_csv('./datasets/dev_semeval_parids-labels.csv')
# %%
# Rebuild train set

rows = []
for idx in range(len(train_ids)):
    par_id = train_ids['par_id'][idx]
    text = df_pcl[df_pcl['par_id'] == par_id]['text'].values[0]
    label = df_pcl[df_pcl['par_id'] == par_id]['label'].values[0]

    rows.append({
        'par_id': par_id,
        'text': text,
        'label': label,
    })
df_train_pcl = pd.DataFrame(rows)
df_train_pcl.to_csv('./datasets/dpm_pcl_train.csv', index=False)

# %%
# Rebuild test set

rows = []
for idx in range(len(test_ids)):
    par_id = test_ids['par_id'][idx]
    text = df_pcl[df_pcl['par_id'] == par_id]['text'].values[0]
    label = df_pcl[df_pcl['par_id'] == par_id]['label'].values[0]

    rows.append({
        'par_id': par_id,
        'text': text,
        'label': label,
    })
df_test_pcl = pd.DataFrame(rows)
df_test_pcl.to_csv('./datasets/dpm_pcl_test.csv', index=False)
# %%
