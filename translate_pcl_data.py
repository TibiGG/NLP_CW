#%%
import pandas as pd
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import torch
import sys
from tqdm import tqdm
from back2back import Back2BackTranslator

tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)

# Get language to translate in from command line
lang = sys.argv[1]


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
train_df = data_pcl.loc[data_pcl.par_id.isin(train_ids.par_id)][['par_id', 'text', 'labels']]

#%%
yes_pcl = train_df.loc[train_df.labels==True]
b2b = Back2BackTranslator()

other_langs = ['pt', 'fr', 'de', 'sw', 'sp', 'th', 'it', 'bg', 'ko']
print('nb yes copies', len(other_langs))

yes_pcl_trans_dict = dict()
print(f'Translating back {lang}')
yes_pcl_trans_dict[lang] = yes_pcl.copy()
yes_pcl_trans_dict[lang]['text'] = \
    yes_pcl['text'].progress_apply(lambda text: b2b.translate_back2back(lang, text))

#%%
yes_pcl_trans_df = pd.concat(yes_pcl_trans_dict.values())

# Save dataframe with translations to a file
yes_pcl_trans_path = f"./datasets/data_pcl_translations_{lang}.csv"
yes_pcl_trans_df.to_csv(yes_pcl_trans_path)