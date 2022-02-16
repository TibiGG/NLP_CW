# %%
import pandas as pd
import re

# %%
pcl_data = pd.read_csv(
    'dontpatronizeme_pcl.tsv',
    sep='\t'
)


# %%
def count_words(text):
    text = text.lower()
    text = re.sub("[^\w ]", "", text)
    words = text.split(" ")
    return len(words)

# %%
def find_problem(text):
    if len(str(text)) < 10:
        return text
    else:
        return -17

# %%
pcl_data['text'] = pcl_data['text'].apply(count_words)

# %%
word_count_per_label = pcl_data[['text', 'pcl']]
wc_group = word_count_per_label.groupby(word_count_per_label['pcl'])
mean_stds = wc_group.aggregate(['mean', 'std'])

# %%
def pcl_label_to_bin(label):
    if label <= 1:
        return 0
    else:
        return 1

# %%
wc_bin = word_count_per_label.copy()
wc_bin['pcl'] = wc_bin['pcl'].apply(pcl_label_to_bin)
wc_bin_group = wc_bin.groupby(wc_bin['pcl'])
bin_mean_stds = wc_bin_group.aggregate(['mean', 'std'])

# %%
print(pcl_data['pcl'].value_counts())
