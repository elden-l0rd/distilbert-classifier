import numpy as np
import random
import pandas as pd
import itertools
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch import cuda
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import words
import sys

pd.set_option('display.max_colwidth', None)

SEED = 64
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    device = 'cuda'
else: 
    device = 'cpu'

PATH = 'data/external/dataset.xlsx'
df = pd.read_excel(PATH)
df = df.dropna(subset=['RAPIDS'])
print(df['RAPIDS'].value_counts())
df = df[df['RAPIDS'].isin([1, 3])]
change_mapping = {1: 0, 3: 1}
df['RAPIDS'] = df['RAPIDS'].map(change_mapping)
# distilBERT model --> 2, 4
# remove 2, 4, 5
df['Desc'] = df['Name'] + ' ' + df['Desc']

# stop_words = set(stopwords.words('english'))

# def remove_stopwords(text):
#     tokens = word_tokenize(text)
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
#     return ' '.join(filtered_tokens)

# df['Desc'] = df['Desc'].apply(remove_stopwords)

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-05
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.Desc = dataframe.Desc
        self.targets = dataframe.RAPIDS
        self.max_len = max_len

    def __len__(self):
        return len(self.Desc)

    def __getitem__(self, index):
        Desc = str(self.Desc[index])
        Desc = " ".join(Desc.split())

        tokens = self.tokenizer.tokenize(Desc)
        if len(tokens) > 510:
            tokens = tokens[:128] + tokens[-382:]
            Desc = self.tokenizer.convert_tokens_to_string(tokens)

        inputs = self.tokenizer.encode_plus(
            Desc,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

train_size = 0.9
train_dataset, test_dataset = train_test_split(df, train_size=train_size, random_state=SEED)
train_dataset = train_dataset.reset_index(drop=True)
test_dataset = test_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = Dataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        distilbert_output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = torch.nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)
        return output

model = DistilBERTClass().to(device)

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        targets = data['targets'].to(device)
        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)

def validation(model, testing_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['targets'].to(device)
            outputs = model(ids, mask)
            outputs = torch.softmax(outputs, dim=1).cpu().numpy()
            pred_labels = np.argmax(outputs, axis=1)
            true_labels.extend(targets.cpu().numpy())
            predictions.extend(pred_labels)
    return predictions, true_labels

predictions, true_labels = validation(model, testing_loader)
cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
f1 = f1_score(true_labels, predictions, average='weighted')
print("Confusion Matrix:\n", cm)
print("F1 score:", f1)

# torch.save(model.state_dict(), 'src/models/BERT/distilBert_RAPIDS.pth')
# model = torch.load('src/models/BERT/distilBert_RAPIDS.pth')

'''
Without removing stopwords:
Epoch: 0, Loss:  0.6883463263511658
Epoch: 1, Loss:  0.2140614092350006
Epoch: 2, Loss:  0.19420138001441956
Confusion Matrix:
 [[17  0]
 [ 7 14]]
F1 score: 0.8130937098844673

After removing stopwords:
Epoch: 0, Loss:  0.6895748972892761
Epoch: 1, Loss:  0.19577600061893463
Epoch: 2, Loss:  0.13437728583812714
Confusion Matrix:
 [[16  1]
 [ 4 17]]
F1 score: 0.8686946055367107
'''

######## UNCOMMENT TO FIND BEST HYPERPARAMETERS ########

# from sklearn.model_selection import ParameterSampler
# param_grid = {
#     'learning_rate': np.logspace(-5, -3, 4),
#     'batch_size': [4, 8, 16],
#     'epochs': [2, 3, 4]
# }

# n_iter = 10

# param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

# # to train and evaluate the model
# def evaluate_model(params):
#     train_loader = DataLoader(training_set, batch_size=params['batch_size'], shuffle=True)
#     model = DistilBERTClass().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
#     for epoch in range(params['epochs']):
#         train(epoch)
#     preds, labels = validation(model, testing_loader)
#     score = f1_score(labels, preds, average='weighted')
#     return score

# best_score = 0
# best_params = {}
# for params in param_list:
#     score = evaluate_model(params)
#     if score > best_score:
#         best_score = score
#         best_params = params

# print(f"Best Score: {best_score}")
# print(f"Best Params: {best_params}")