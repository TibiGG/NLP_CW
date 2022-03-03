
import pandas as pd
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from back2back import Back2BackTranslator, translate

tqdm.pandas()

other_langs = ['pt', 'fr', 'de', 'sw', 'sp', 'it', 'bg', 'ko']

translated_dataset = dict()
for lang in other_langs:
    translated_dataset[lang] = pd.read_csv(f"./datasets/data_pcl_translations_{lang}.csv")[['par_id', 'text']]

translated_dataset_df = pd.concat(translated_dataset.values())
print(len(translated_dataset_df))

yes_pcl_trans_path = "./datasets/data_pcl_translations.csv"
translated_dataset_df.to_csv(yes_pcl_trans_path)