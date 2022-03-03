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
if device == "cuda":
    print("Emptying Cache")
    torch.cuda.empty_cache()


#%%
train_ids = pd.read_csv('./datasets/train_semeval_parids-labels.csv')
test_ids = pd.read_csv('./datasets/dev_semeval_parids-labels.csv')
data_pcl = pd.read_csv("./datasets/dontpatronizeme_pcl.tsv", sep="\t", skiprows=3,
                        names=['par_id','art_id','keyword','country_code','text','label'])


#%%
# Binary labels
data_pcl['labels'] = data_pcl.label >= 2


#%%
# Seperate train and test df according to train_ids and test_ids
train_df = data_pcl.loc[data_pcl.par_id.isin(train_ids.par_id)][['par_id','text', 'labels']]
test_df = data_pcl.loc[data_pcl.par_id.isin(test_ids.par_id)][['par_id','text', 'labels']]

yes_pcl = train_df.loc[train_df.labels==True]
no_pcl = train_df.loc[train_df.labels==False]

# Augment yes_pcl data frame with backtranslated data

#%%
yes_pcl_translations = pd.read_csv("./datasets/data_pcl_translations.csv")
yes_pcl_translations['labels'] = True
yes_pcl = pd.concat([yes_pcl, yes_pcl_translations])
new_train = pd.concat([yes_pcl, no_pcl])[['text', 'labels']]
print('nb yes', (new_train['labels'] > .5).sum())
print('nb no', (new_train['labels'] < .5).sum())

#%%
n_examples = len(new_train)
print('nb examples', n_examples)
new_train = new_train.iloc[np.random.permutation(n_examples)]


task1_model_args = ClassificationArgs(num_train_epochs=5,
                                        no_save=False,
                                        no_cache=False,
                                        overwrite_output_dir=True,
                                        evaluate_during_training=True, 
                                        output_dir='./outputs/outputs_roberta_yes_copies', #by default
                                        best_model_dir='./outputs/outputs_roberta_yes_copies/best_model',
                                        max_seq_length=512, #by default 128, it could be intresting to see if this trucates our texts
                                        save_eval_checkpoints=False,
                                        save_model_every_epoch=True,
                                        save_steps=-1,
                                        evaluate_during_training_verbose=False,
                                        learning_rate=4e-5,
                                        train_batch_size=8,
                                        )


task1_model = ClassificationModel("roberta", "roberta-base",
                                    args=task1_model_args,
                                    use_cuda=torch.cuda.is_available()
                                    )


# Test code on a smaller portion of the dataset
#mini_train_df = train_df.loc[[0,1,2,3,4,10423,10444,10453,10466,10468]]
#mini_test_df = test_df.loc[[150,153,10462,10463]]

# Run the modelb
task1_model.train_model(new_train, show_running_loss=True, eval_df=test_df, f1=f1_score)
    