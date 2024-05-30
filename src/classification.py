'''
Classification using DistilBERT

Loads a trained DistilBERT model from directory and uses it to classify a dataset.
Note: Ensure that column names are correct when using dataset.
'''

import numpy as np
import pandas as pd
import random
import joblib
import torch
import transformers
from transformers import DistilBertTokenizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from modules import preprocessing as pre

'''
Each category is represented by a number:
S   T   I   D   E
|   |   |   |   |
0   1   2   3   4
'''

if __name__ == '__main__':
    SEED = 64
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        device = 'cuda'
    else: device = 'cpu'

    df = pd.read_excel('data/external/raw/mitre-attack-framework.xlsx', sheet_name='Threats')
    df["Desc"] = df["Name"] + " " + df["Desc"]
    
    class DistilBERTClass(torch.nn.Module):
        def __init__(self):
            super(DistilBERTClass, self).__init__()
            self.l1 = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.pre_classifier = torch.nn.Linear(768, 768)
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(768, 5) # 5 classes

        def forward(self, input_ids, attention_mask):
            distilbert_output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = distilbert_output[0]
            pooled_output = hidden_state[:, 0]
            pooled_output = self.pre_classifier(pooled_output)
            pooled_output = torch.nn.ReLU()(pooled_output)
            pooled_output = self.dropout(pooled_output)
            output = self.classifier(pooled_output)
            return output

    model = DistilBERTClass()
    model.load_state_dict(torch.load('src/models/BERT/distilBert_STRIDE.pth'))
    model.to(device)
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def predict_class(sentence):
        tokens = tokenizer.tokenize(sentence)

        if len(tokens) > 512:
            tokens = tokens[:128] + tokens[-382:]
            sentence = tokenizer.convert_tokens_to_string(tokens)

        tokens = tokenizer(sentence, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.item()

    tokens = tokenizer(df['Desc'].apply(str).tolist(), max_length=512, pad_to_max_length=True, truncation=True, return_tensors="pt")

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)
    predictions = predictions.cpu()

    df['predicted_label'] = predictions
    accuracies = df.groupby('label').apply(lambda group: (group['predicted_label'] == group['label']).sum() / len(group))
    print("Accuracy per label:")
    for label, acc in accuracies.items():
        total_samples = len(df[df['label'] == label])
        correct_classifications = (df[df['label'] == label]['predicted_label'] == label).sum()
        print(f"Label {label}: {correct_classifications}/{total_samples} correctly classified. Accuracy: {acc:.2%}")
