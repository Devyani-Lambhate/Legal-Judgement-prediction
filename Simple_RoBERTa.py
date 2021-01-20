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

CUDA_LAUNCH_BLOCKING=1

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

df=pd.DataFrame({'text':train_x['TEXT'],'labels':train_x['BINARY_CONCLUSION']})
df_train,df_val = df[0:train_length],df[train_length:train_length+test_length]

from simpletransformers.classification import ClassificationModel

model = ClassificationModel('roberta', 'roberta-base', num_labels=5,args = {'max_seq_length':512,'reprocess_input_data': True, 'overwrite_output_dir': True})
model.train_model(df_train)
result, model_outputs, wrong_predictions = model.eval_model(df_val)

from scipy.special import softmax
y_pred = softmax(model_outputs,axis=1).argmax(axis=1)
y_true = df_val.labels.to_numpy()


print(' test precision recall fscore', precision_recall_fscore_support(y_true, y_pred, average='macro'))

