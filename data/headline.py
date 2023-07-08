import json
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

headlines = pd.DataFrame(columns=['headline', 'is_sarcastic'])

# assign content to each columnd from the dataset
text = []
labels = []

with open('/Sarcasm_Headlines_Dataset.json') as file:
    for line in file:
        data_json = json.loads(line)
        text.append(data_json['headline'])
        labels.append(data_json['is_sarcastic'])

headlines['headline'] = text
headlines['is_sarcastic'] = labels

# shuffle row of dataframe
headlines = headlines.dropna()
headlines = headlines.sample(frac=1).reset_index(drop=True)

# TODO - train_test_split (80/20 split)

# TODO - tokenize dataset