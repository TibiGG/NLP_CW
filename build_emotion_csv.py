#!/usr/bin/env python3
# %%

from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm.notebook import trange

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", truncation=True, max_length=512)
df_pcl = pd.read_csv('./datasets/dpm_pcl_train.csv')

# %%

emotions = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

rows = []
for idx in trange(len(df_pcl)):
    par_id = df_pcl["par_id"][idx]
    text = df_pcl["text"][idx]


    sentiment = classifier(text)[0]["label"]
    rows.append({
        "par_id": par_id,
        "text": text,
        "label": emotions[sentiment]
    })

df_emotion = pd.DataFrame(rows)
# %%
df_emotion.to_csv("./datasets/dpm_pcl_emotion_train.csv", index=False)
# %%
