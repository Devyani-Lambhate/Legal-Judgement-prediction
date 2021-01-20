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


path_train='/home/devyani/NLP Project/ECHR_Dataset/EN_train/'
path_test='/home/devyani/NLP Project/ECHR_Dataset/EN_test/'

def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1

def load_data(subset):
	DATASET=[]
	if(subset=='train'):
		path=path_train

	if(subset=='test'):
		path=path_test

	for fname in sorted(os.listdir(path)):
			with open(path+fname, "r") as read_file:
				data = json.load(read_file)
				DATASET.append(data)

	x = pd.DataFrame(data=DATASET)

	BINARY_CONCLUSION=[]
	for item in x['VIOLATED_ARTICLES']:
			#print(item)
			if(item==[]):
				BINARY_CONCLUSION.append('0')
			else:
				BINARY_CONCLUSION.append('1')

	x.insert(16, "BINARY_CONCLUSION", BINARY_CONCLUSION, True)
	

	text_arr=[]

	for t in x['TEXT']:
		text_arr.append(listToString(t))

	target_arr=[]

	for b in x['BINARY_CONCLUSION']:
		target_arr.append(listToString(b))



	data={ 'data': text_arr,
	       'target': target_arr
	}

	return data
