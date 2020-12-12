# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:56:52 2020

@author: HP
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import LabelEncoder


# primer modelo
def td(x, y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1) #hiperparametros originales
    #clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=5, min_samples_leaf=20) #hiperparametros afinados
    clf = DecisionTreeClassifier(criterion="gini", max_depth=4, max_features="auto", random_state=5, min_samples_leaf=20)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print("------Decision Tree-------")
    print(clf.score(x_test, y_test))
    print("----------matriz de confusión---------")
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    print("-----------cross validation---------")
    scores=cross_val_score(clf,x,y,cv=10)
    print(scores)
    print("promedio: ", scores.mean())
    print("desviación estándar: ", scores.std()*2)
    print("----------métricas---------")
    print(classification_report(y_test, y_pred))
    
    # glosario para el afinamiento de hiperparametros
    
    # param_grid = {'min_samples_leaf': [20,40], 'max_features': ['auto', 'sqrt', 'log2'],
    #               'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy']}
    # grid=GridSearchCV(clf, param_grid, refit=True, verbose=2)
    # grid.fit(x_train,y_train)
    # print(grid.best_estimator_)
    
    
# segundo modelo    
def rf(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    #clf = RandomForestClassifier(n_estimators=1000, random_state=0, warm_start=True,oob_score=True)#original
    clf = RandomForestClassifier(n_estimators=200, random_state=0, warm_start=True,oob_score=True, criterion= 'entropy', max_depth=7, max_features="auto")#hiperparametros afinados
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    print("------Random Forest-------")
    print(clf.score(x_test, y_test))
    print("----------matriz de confusión---------")
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    print("-----------cross validation---------")
    scores=cross_val_score(clf,x,y,cv=10)
    print(scores)
    print("promedio: ", scores.mean())
    print("desviación estándar: ", scores.std()*2)
    print("----------métricas---------")
    print(classification_report(y_test, y_pred))
    
    # glosario para el afinamiento de hiperparametros
    
    # param_grid = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'],
    #               'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy']}
    # grid=GridSearchCV(clf, param_grid, refit=True, verbose=2)
    # grid.fit(x_train,y_train)
    # print(grid.best_estimator_)
    
    
#función para preprocesar la data    
def labelEncoder(df, obj):
    label_enc = LabelEncoder()
    for i in obj:
        df[i] = label_enc.fit_transform(df[i].astype(str))
    return df

#reemplaza los espacios con ?
def reemplazar(df):
    df.replace('?',-9999,inplace=True)
    return df 

def main():
    pd.set_option('display.max_rows', 10)
    df = pd.read_csv("chronic_kidney_disease.txt")
    columnNames = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe", "ane", "class"]
    df.columns = columnNames
    
    # for i in range (0,399):
    #     if (df["class"][i] != 'ckd' and df["class"][i] != 'notckd'):
    #         print(df["class"][i])
    #         print(i)
    
    
    # for i in range(len(df)):
    #     # sfile = df.values[i][0]
    #     # dst = df.values[i][1]
    #     for x in range(15):
    #         print(type(x))
    
    
    
    df = reemplazar(df)
    # print(df)
    #Te saca todas las columnas que sean de string
    objetos = df.select_dtypes(include = "object").columns
    # print(objetos)
    #Va a cambiar todos los valores a numericos
    df = labelEncoder(df, objetos)
    # print(df)
    
    # print(df[["appet"]])
    
    
    y = df["class"]
    listDrop = ["class"]
    x = df.drop(listDrop, axis=1)
    
    
    # print(x)
    # print(y)
    
    rf(x,y)
    td(x,y)
    
    


if __name__=="__main__":
    main()