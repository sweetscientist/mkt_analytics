# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:50:44 2020

@author: lilph
"""


# Banking Marketing Analysis
import sklearn
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
#import pylab as plb
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
%matplotlib inline

# Import the data as CSV
data = pd.read_csv('marketing_data.csv')

# Replace word
data.replace(['basic.6y', 'basic.4y', 'basic.9y'], 'basic', inplace=True)

# Summary of numerical Data
summary = data.describe()

plt.figure(figsize=(8,4))
#### Visualization
# View count of column y AKA Subscribed a term deposit(binary)
sns.countplot(x='y', data=data).set_title('Not Subscribed V.S. Subscribed')

# View count of column job
sns.countplot(y='job', data=data, order=data['job'].value_counts().index).set_title('Job Distribution')

# Marital status
sns.countplot(x='marital', data=data).set_title('Marital Status')

# Education
sns.countplot(y='education', data=data, order=data['education'].value_counts().index).set_title('Education Levels')

# Default
sns.countplot(x='default',data=data, order=data['default'].value_counts().index).set_title('Default Count')

# Housing
sns.countplot(x='housing',data=data, order=data['housing'].value_counts().index).set_title('Housing Status')

# Loan
sns.countplot(x='loan', data=data).set_title('Loan Count')

# Contact
sns.countplot(x='contact', data=data).set_title('Method of Contact')

# Campaign
sns.countplot(y='campaign', data=data).set_title('Campaign Status')

# Pdays excluding 999
sns.countplot(x='pdays', data=data[data.pdays!=999]).set_title('Padys Count')

# Previous
sns.countplot(x='previous', data=data).set_title('Previous Count')

# Poutcome
sns.countplot(x='poutcome', data=data).set_title('Poutcome Count')

# Emp Var Rate
sns.countplot(x='emp_var_rate', data=data, order=data['emp_var_rate'].value_counts().index).set_title('Emp Var Rate Distribution')

# Cons price index
sns.distplot(data.cons_price_idx).set_title('Con Price Index Distribution')

# Cons conf index
sns.distplot(data.cons_conf_idx).set_title('Con Conf Index Distribution')

# Euribor3m
sns.countplot(x='euribor3m', data=data).set_title('Euribor3m')
'euribor3m', 'nr_employed'

if __name__=="__main__": 
    data = pd.read_csv('marketing_data.csv')
    print(data.describe())
    print(data.isnull().sum())
