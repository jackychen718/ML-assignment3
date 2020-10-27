import numpy as np
from statistics import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_breast_cancer():
    handle = open('breast-cancer-wisconsin.data', 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[i for i in r.split(',')] for r in rows if r])
    out=out[:,1:]
    for col in range(9):
        out[:,col][out[:,col]=='?']=mode(out[:,col])
    np.random.seed(100)
    np.random.shuffle(out)
    train_features=np.array(out[:,:-1],dtype=float)
    train_features_norm=train_features/train_features.max(axis=0)
    train_labels=np.array(out[:,-1],dtype=int)
    train_labels[train_labels==4]=1
    train_labels[train_labels==2]=0
    return train_features_norm[:199],train_labels[:199],train_features_norm[199:],train_labels[199:]
    
def read_spam(my_seed=100):
    handle = open('spambase.data', 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[i for i in r.split(',')] for r in rows if r])
    np.random.seed(my_seed)
    np.random.shuffle(out)
    train_features=np.array(out[:,:-1],dtype=float)
    train_labels=np.array(out[:,-1],dtype=int)
    train_features_norm=train_features/train_features.max(axis=0)
    return train_features_norm[:601],train_labels[:601],train_features_norm[601:],train_labels[601:]