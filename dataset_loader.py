import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import arff
sns.set()
#Loading the datasets
cc=pd.read_csv('datasets/credit_card_data.csv') 
wq=pd.read_csv('datasets/winequality.csv')
audit=pd.read_csv('datasets/audit_risk.csv')
diabetes=pd.read_csv('datasets/diabetes.csv')
news=pd.read_csv('datasets/OnlineNewsPopularity_fyp.csv')
bn=pd.read_csv('datasets/data_banknote_authentication.csv')
ss_train=arff.loadarff('datasets/SmoothSubspace_TRAIN.arff')
ss_test=arff.loadarff('datasets/SmoothSubspace_TEST.arff')
ct_train=arff.loadarff('datasets/Chinatown_TRAIN.arff')
ct_test=arff.loadarff('datasets/Chinatown_TEST.arff')
sd=pd.read_csv('datasets/seeds_dataset.csv')
x=news["shares"].mean()
maxi=news["shares"].max()
bins=[0,x,maxi]
news['shares'] = pd.cut(news['shares'], bins,labels=["0","1"]) #Converting the regression dataset into a classification dataset and binning all the classes into two classes.
del news['url'] #Removing irrelevant columns from the dataset
row=news['shares']
row=row.tolist()
del news['shares']
news['shares']=row
del cc['ID']
