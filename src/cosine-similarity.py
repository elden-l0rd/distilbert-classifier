'''
########### Cosine similarity ###########
Compares the cosine similarity between the embeddings of the descriptions in the dataset.

Intra-label compares the similarity between descriptions of the same label.
Inter-label compares the similarity between descriptions of different labels.

Results are at the end of the code.
'''

import numpy as np
import pandas as pd
import itertools
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

PATH = 'data/external/raw/raw_capec_data_BERT.xlsx'
df = pd.read_excel(PATH)
label_changes = {3: 2, 4: 3, 5: 4}
df['label'] = df['label'].replace(label_changes)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def encode_texts(texts):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=1200)
    with torch.no_grad():
        features = model(**encoded)
    return features.last_hidden_state.mean(dim=1).numpy()

def intra_label(df):
    results = []
    for label in df['label'].unique():
        label_data = df[df['label'] == label]['Embeddings'].tolist()
        if len(label_data) > 1:
            sim_matrix = cosine_similarity(label_data)
            avg_sim = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
            results.append(avg_sim)
        else:
            results.append(None)
    return results

def inter_label(df):
    results = []
    labels = sorted(df['label'].unique())
    for l1, l2, in itertools.combinations(labels, 2):
        d1 = df[df['label'] == l1]['Embeddings'].tolist()
        d2 = df[df['label'] == l2]['Embeddings'].tolist()
        if d1 and d2:
            sim_matrix = cosine_similarity(d1, d2)  
            avg_sim = np.mean(sim_matrix)
            results.append(avg_sim)
        else:
            results.append(None)
    return results

embeddings = encode_texts(df['Desc'].astype(str).tolist())
df['Embeddings'] = list(embeddings)
intra_similarities = intra_label(df)
inter_similarities = inter_label(df)

intra_stats = {
    'Average': np.mean(intra_similarities),
    'Mean': np.mean(intra_similarities),
    'Median': np.median(intra_similarities),
    'Min': np.min(intra_similarities),
    'Max': np.max(intra_similarities)
}
print("Intra-label Cosine Similarities:")
for key, value in intra_stats.items():
    print(f"{key}: {value:.3f}")
    

inter_stats = {
    'Average': np.mean(inter_similarities),
    'Mean': np.mean(inter_similarities),
    'Median': np.median(inter_similarities),
    'Min': np.min(inter_similarities),
    'Max': np.max(inter_similarities)
}
print("\nInter-label Cosine Similarities:")
for key, value in inter_stats.items():
    print(f"{key}: {value:.3f}")


'''
Results for Intra-label Cosine Similarities:
Average: 0.880
Mean: 0.880
Median: 0.883
Min: 0.854
Max: 0.892

Results for Inter-label Cosine Similarities:
Average: 0.868
Mean: 0.868
Median: 0.873
Min: 0.853
Max: 0.882
'''
