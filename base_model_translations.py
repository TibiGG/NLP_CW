#%%
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import torch
from collections import Counter
from tqdm import tqdm
tqdm.pandas()

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
if device != "cpu":
    print("Emptying Cache")
    torch.cuda.empty_cache()


#%%
train_ids = pd.read_csv('./datasets/train_semeval_parids-labels.csv')
test_ids = pd.read_csv('./datasets/dev_semeval_parids-labels.csv')
data_pcl = pd.read_csv("./datasets/dontpatronizeme_pcl.tsv", sep="\t", skiprows=3,
                        names=['par_id','art_id','keyword','country_code','text','label'])


#%%
# Binary labels
data_pcl['is_pcl'] = data_pcl.label >= 2


#%%
# Seperate train and test df according to train_ids and test_ids
train_df = data_pcl.loc[data_pcl.par_id.isin(train_ids.par_id)][['par_id','keyword','text', 'is_pcl']]
test_df = data_pcl.loc[data_pcl.par_id.isin(test_ids.par_id)][['par_id','text', 'is_pcl']]

yes_pcl = train_df.loc[train_df.is_pcl==True]
no_pcl = train_df.loc[train_df.is_pcl==False]

#%%
# Separate data frame into train and validation sets
def separate_train_validation(pcl_df, percent_validation=0.10):
    assert(0 <= percent_validation <= 1)
    keywords = set(train_df.keyword.to_list())
    pcl_validation_list = []
    pcl_train_list = []
    for keyword in keywords:
        pcl_with_keyword = pcl_df.loc[pcl_df.keyword==keyword]
        validation_set_len = int(np.floor(len(pcl_with_keyword) * percent_validation))
        pcl_validation_list.append(pcl_with_keyword[:validation_set_len])
        pcl_train_list.append(pcl_with_keyword[validation_set_len:])

    pcl_validation_df = pd.concat(pcl_validation_list)
    pcl_train_df = pd.concat(pcl_train_list)

    return pcl_validation_df, pcl_train_df 

yes_pcl_validation, yes_pcl_train = separate_train_validation(yes_pcl)
no_pcl_validation, no_pcl_train = separate_train_validation(no_pcl)

#%%
# Augment yes_pcl data frame with backtranslated data
def augment_with_translations(yes_pcl_df, no_pcl_df):
    yes_pcl_translations = pd.read_csv("./datasets/data_pcl_translations.csv")
    yes_pcl_translations['is_pcl'] = True
    yes_pcl_translations = yes_pcl_translations.loc[yes_pcl_translations.par_id.isin(yes_pcl_df.par_id)]
    yes_pcl_df = pd.concat([yes_pcl_df, yes_pcl_translations])
    new_df = pd.concat([yes_pcl_df, no_pcl_df])[['text', 'is_pcl']]
    print('nb yes', (new_df['is_pcl'] > .5).sum())
    print('nb no', (new_df['is_pcl'] < .5).sum())
    return new_df

#%%
new_validation = augment_with_translations(yes_pcl_validation, no_pcl_validation)
new_train = augment_with_translations(yes_pcl_train, no_pcl_train)
print('nb yes validation', (new_validation['is_pcl'] > .5).sum())
print('nb no validation', (new_validation['is_pcl'] < .5).sum())
print('nb yes train', (new_train['is_pcl'] > .5).sum())
print('nb no train', (new_train['is_pcl'] < .5).sum())

#%%
n_examples = len(new_train)
print('nb examples', n_examples)
new_train = new_train.iloc[np.random.permutation(n_examples)]

batch_size = 16
seq_length = 128

task1_model_args = ClassificationArgs(num_train_epochs=1000,
                                      no_save=False,
                                      no_cache=False,
                                      overwrite_output_dir=True,
                                      evaluate_during_training=True,
                                      output_dir=f'./outputs/roberta_large_bs_{batch_size}_seq_{seq_length}', #by default
                                      best_model_dir=f'./outputs/roberta_large_bs_{batch_size}_seq_{seq_length}/best_model',
                                      max_seq_length=seq_length, #by default 128, it could be intresting to see if this trucates our texts
                                      save_eval_checkpoints=False,
                                      save_model_every_epoch=True,
                                      save_steps=-1,
                                      evaluate_during_training_verbose=False,
                                      learning_rate=4e-5,
                                      train_batch_size=batch_size,
                                      early_stopping_metric='f1',
                                      early_stopping_metric_minimize=False,
                                      early_stopping_patience=100,
                                      )


task1_model = ClassificationModel("roberta", "roberta-large",
                                    args=task1_model_args,
                                    use_cuda=torch.cuda.is_available()
                                    )


# Test code on a smaller portion of the dataset
#mini_train_df = train_df.loc[[0,1,2,3,4,10423,10444,10453,10466,10468]]
#mini_test_df = test_df.loc[[150,153,10462,10463]]

# Run the model
task1_model.train_model(new_train, show_running_loss=True, eval_df=new_validation, f1=f1_score)
    