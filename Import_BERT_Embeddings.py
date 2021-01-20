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

def BERT_embeddings():

		DATASET=[]
		for fname in sorted(os.listdir(path)):
			with open(path+fname, "r") as read_file:
				data = json.load(read_file)
				DATASET.append(data)
		n=len(DATASET)
		print(n)
		for fname in sorted(os.listdir(path_test)):
			with open(path_test+fname, "r") as read_file:
				data = json.load(read_file)
				DATASET.append(data)
				#print(data)

		n=len(DATASET)
		print(n)
		train_x = pd.DataFrame(data=DATASET)
		print(train_x.columns)

		#print(train_x['TEXT'][0])
		for i in range(n):
			listToStr = ' '.join([str(elem) for elem in train_x['TEXT'][i]])
			train_x['TEXT'][i]= nltk.tokenize.sent_tokenize(listToStr)



		print(train_x['TEXT'])
		print(train_x['CONCLUSION'])

		def check(string, sub_str): 
		    if (string.find(sub_str) == -1): 
		        return "NO"
		    else: 
		        return "YES" 

		BINARY_CONCLUSION=[]
		for item in train_x['CONCLUSION']:
			sub_str="Violation of Art."
			BINARY_CONCLUSION.append(check(item, sub_str))

		train_x.insert(16, "BINARY_CONCLUSION", BINARY_CONCLUSION, True) 

		print(train_x['BINARY_CONCLUSION'])



		############## LOAD THE MODEL#####################################

		# For DistilBERT:
		model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

		## Want BERT instead of distilBERT? Uncomment the following line:
		#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

		# Load pretrained model/tokenizer
		tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
		model = model_class.from_pretrained(pretrained_weights)

		f_features=[]
		for i in range(n):
			print('----------',i,'---------')
			df = pd.DataFrame(train_x['TEXT'][i])
			tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True,max_length=512)))
			max_len = 512

			padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
			attention_mask = np.where(padded != 0, 1, 0)
			attention_mask.shape

			input_ids = torch.tensor(padded)  
			attention_mask = torch.tensor(attention_mask)

			print(attention_mask.shape)

			with torch.no_grad():
			    last_hidden_states = model(input_ids, attention_mask=attention_mask)

			features = last_hidden_states[0][:,0,:].numpy()
			features=np.sum(features, axis=0)
			f_features.append(features)
			

		labels=train_x['BINARY_CONCLUSION']
		f_features=np.array(f_features)
		print(f_features.shape)
		print(labels.shape)

		train_features=f_features[0:7100,:]
		test_features=f_features[7100:10098,:]
		train_labels=labels[0:7100]
		test_labels = labels[7100:10098]

		print(train_features.shape)
		print(test_features.shape)
		print(train_labels.shape)
		print(test_labels.shape)

		return train_features,test_features,train_labels,test_labels


####################### ATTENTION ########################


###########################################################

#################FITTING LINEAR CLASSIFIER ON TOP OF BERT########################

# parameters = {'C': np.linspace(0.0001, 100, 20)}
# grid_search = GridSearchCV(LogisticRegression(), parameters)
# grid_search.fit(train_features, train_labels)

# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)
from sklearn.metrics import precision_recall_fscore_support


lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
print('test score',lr_clf.score(test_features, test_labels))
print('train score',lr_clf.score(train_features, train_labels))
y_true=test_labels
y_pred=lr_clf.predict(test_features)

print(precision_recall_fscore_support(y_true, y_pred))


  
# To show the plot 
plt.show() 
