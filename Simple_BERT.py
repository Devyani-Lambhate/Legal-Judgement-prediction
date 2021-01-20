import os
import json
# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import torch
# Preliminaries
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
# Models
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
# Training
import torch.optim as optim
# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')

import numpy as np

path='ECHR_Dataset/EN_train/'
path_test='ECHR_Dataset/EN_test/'

def prepare_data():

		DATASET=[]
		for fname in sorted(os.listdir(path)):
			with open(path+fname, "r") as read_file:
				data = json.load(read_file)
				DATASET.append(data)

		x = pd.DataFrame(data=DATASET)
		#x = x[x['CONCLUSION'] !="Inadmissible"]
		train_length=len(x)
		print(train_length)
		for fname in sorted(os.listdir(path_test)):
			with open(path_test+fname, "r") as read_file:
				data = json.load(read_file)
				DATASET.append(data)
				#print(data)

		x = pd.DataFrame(data=DATASET)
		#x = x[x['CONCLUSION'] !="Inadmissible"]

		test_length=len(x)-train_length
		print(test_length)

		train_x = pd.DataFrame(data=DATASET)
		#train_x = train_x[train_x['CONCLUSION'] !="Inadmissible"]
		print(train_x.columns)

		#print(train_x['TEXT'][0])
		#for i in range(n2):
		#	listToStr = ' '.join([str(elem) for elem in train_x['TEXT'][i]])
		#	train_x['TEXT'][i]= nltk.tokenize.sent_tokenize(listToStr)



		print(train_x['TEXT'])
		print(train_x['VIOLATED_ARTICLES'])

		def check(string, sub_str): 
		    if (string.find(sub_str) == -1): 
		        return 0
		    else: 
		        return 1 

		BINARY_CONCLUSION=[]
		for item in train_x['VIOLATED_ARTICLES']:
			#print(item)
			if(item==[]):
				BINARY_CONCLUSION.append(0)
			else:
				BINARY_CONCLUSION.append(1)






			
		train_x.insert(16, "BINARY_CONCLUSION", BINARY_CONCLUSION, True)

		print(train_x['BINARY_CONCLUSION'].value_counts())
		

		return train_x,train_length,test_length


train_x,train_length,test_length=prepare_data()

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


tokenized = train_x['TEXT'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512)))
print(tokenized)
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()

labels = train_x['BINARY_CONCLUSION']

train_features=features[0:train_length]
test_features=features[train_length:train_length+test_length]

train_labels=labels[0:train_length]
test_labels = labels[train_length:train_length+test_length]

# parameters = {'C': np.linspace(0.0001, 100, 20)}
# grid_search = GridSearchCV(LogisticRegression(), parameters)
# grid_search.fit(train_features, train_labels)

# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
from sklearn.metrics import precision_recall_fscore_support
print('test score',lr_clf.score(test_features, test_labels))

y_true=test_labels
y_pred=lr_clf.predict(test_features)
print(y_true,y_pred)
print(' test precision recall fscore', precision_recall_fscore_support(y_true, y_pred, average='macro'))
y_true=train_labels
y_pred=lr_clf.predict(train_features)
print(' train precision recall fscore', precision_recall_fscore_support(y_true, y_pred, average='macro'))

