#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[16]:


import seaborn as sns;
import matplotlib.pyplot as plt
import pandas as pd;
import numpy as np;
import streamlit as st;
from joblib import dump, load
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import cross_val_score;
from sklearn.preprocessing import LabelEncoder;
from sklearn.ensemble import RandomForestClassifier

# pipeline elements
from sklearn.decomposition import PCA # PCA = Principal Component Analysis
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.ensemble import GradientBoostingClassifier

# pipeline materiaux
from sklearn.pipeline import Pipeline # PCA = Principal Component Analysis
from sklearn.model_selection import GridSearchCV

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from stop_words import get_stop_words
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_curve, auc

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import dump, load


# ## Liens :

# Télécharger le dataset : 
#    <a href="https://drive.google.com/file/d/1OEWrVjE7B2d23-eQrTMcJpRJMxoB6iUH/view?usp=sharing ">labels.csv</a>

# ## Fonction

# In[15]:


app = FastAPI()


# In[94]:


# Créer une mesure de performance

def accuracy(preds, target):
    M = target.shape[0] # Nombre d'exemple
    total_correctes = (preds == target).sum()
    accuracy = total_correctes / M
    return accuracy


# In[95]:


def compare(list):
    tmp = 0
    proba = 0
    for i in range(len(list)):
        if list[i] > list[tmp]:
            tmp = i
            proba = list[i]
    return tmp, proba

def myClasse(classT, proba):
    print(" Classe : %d avec %f %%. " % (classT, proba))
    st.success(" Classe : %d avec %f %%. " % (classT, proba))
    return


# In[13]:


def kn(number, X_tr, X_te, Y_tr, Y_te):
    knn = KNN(n_neighbors=21,
             weights='uniform',
             leaf_size=3)
    knn.fit(X_tr, Y_tr)
    train_preds = knn.predict(X_tr)
    predictions = knn.predict(X_te)
    knnTr = accuracy(train_preds, Y_tr)
    knnTe= accuracy(predictions, Y_te)
    return knnTr, knnTr
    
def rf(number, X_tr, X_te, Y_tr, Y_te):
    clf = RandomForestClassifier(random_state=0,      # Lecture aléatoire des données
                                n_estimators=21,      # Il va diviser X_tr en plusieurs arbres
                                max_depth=7)          # Il va spliter/augmenter la profondeur des arbres
    clf.fit(X_tr, Y_tr)
    train_preds = clf.predict(X_tr)
    predictions = clf.predict(X_te)
    clfTr = accuracy(train_preds, Y_tr)
    clfTe= accuracy(predictions, Y_te)
    return clfTr, clfTe
    
def gb(number, X_tr, X_te, Y_tr, Y_te):
    grad = GradientBoostingClassifier(random_state=60,        # Lecture aléatoire des données
                                     n_estimators=40,         # Nombre d'étape (modèle), plus il y a de modèle, plus il apprendra
                                     max_depth=10)            # Profondeur de l'arborescence
    grad.fit(X_tr, Y_tr)
    train_preds = grad.predict(X_tr)
    predictions = grad.predict(X_te)
    gradTr = accuracy(train_preds, Y_tr)
    gradTe= accuracy(predictions, Y_te)
    return gradTr, gradTe


# ## Collecte de données

# In[133]:


# On récupère notre Dataset et la stocke dans une variable

accident = pd.read_csv('data/US_Accidents_June20_mini.csv')


# In[134]:


# On supprime les colonnes qui nous intéressent pas

accident = accident.drop(['ID'],axis=1)
accident = accident.drop(['Start_Lat'],axis=1)
accident = accident.drop(['Start_Lng'],axis=1)
accident = accident.drop(['End_Lat'],axis=1)
accident = accident.drop(['End_Lng'],axis=1)
accident = accident.drop(['Description'],axis=1)
accident = accident.drop(['Number'],axis=1)
accident = accident.drop(['Street'],axis=1)
accident = accident.drop(['Side'],axis=1)
accident = accident.drop(['Zipcode'],axis=1)
accident = accident.drop(['Country'],axis=1)
accident = accident.drop(['Timezone'],axis=1)
accident = accident.drop(['Weather_Timestamp'],axis=1)
accident = accident.drop(['Wind_Direction'],axis=1)
accident = accident.drop(['Wind_Chill(F)'],axis=1)


# In[135]:


# On enlève toutes les valeurs NaN

accident = accident.dropna(how='any')


# In[100]:


# On conserve notre Dataset sans transformation

accidentNoTransform = accident.copy()


# In[101]:


# On convertie les colonnes dans le type qui nous interessent

accident['Source'] = LabelEncoder().fit_transform(accident['Source'])

accident.TMC = accident['TMC'].astype('category').cat.codes

accident.City = accident['City'].astype('category').cat.codes

accident.State = accident['State'].astype('category').cat.codes

accident.County = accident['County'].astype('category').cat.codes

accident.Airport_Code = accident['Airport_Code'].astype('category').cat.codes

accident.Sunrise_Sunset =accident['Sunrise_Sunset'].astype('category').cat.codes 

accident.Weather_Condition = accident['Weather_Condition'].astype('category').cat.codes

accident.Nautical_Twilight = accident['Nautical_Twilight'].astype('category').cat.codes

accident.Astronomical_Twilight = accident['Astronomical_Twilight'].astype('category').cat.codes

accident.Civil_Twilight = accident['Civil_Twilight'].astype('category').cat.codes

accident['Amenity'] = LabelEncoder().fit_transform(accident['Amenity'])
accident['Bump'] = LabelEncoder().fit_transform(accident['Bump'])
accident['Crossing'] = LabelEncoder().fit_transform(accident['Crossing'])
accident['Give_Way'] = LabelEncoder().fit_transform(accident['Give_Way'])
accident['Junction'] = LabelEncoder().fit_transform(accident['Junction'])
accident['No_Exit'] = LabelEncoder().fit_transform(accident['No_Exit'])
accident['Railway'] = LabelEncoder().fit_transform(accident['Railway'])
accident['Roundabout'] = LabelEncoder().fit_transform(accident['Roundabout'])
accident['Station'] = LabelEncoder().fit_transform(accident['Station'])
accident['Stop'] = LabelEncoder().fit_transform(accident['Stop'])
accident['Traffic_Calming'] = LabelEncoder().fit_transform(accident['Traffic_Calming'])
accident['Traffic_Signal'] = LabelEncoder().fit_transform(accident['Traffic_Signal'])
accident['Turning_Loop'] = LabelEncoder().fit_transform(accident['Turning_Loop'])


# In[102]:


# On va séparer notre target de nos colonnes

Y = accident['Severity'].astype('category').cat.codes # La target va être la gravité
X = accident.drop('Severity', axis='columns') # En fonction des critère environnant, on va essayer de prédir le niveau de gravité


# In[22]:


# On convertie nos column date de type object en type date

X['Start_Time'] = pd.to_datetime(X['Start_Time'], 
 format = '%Y-%m-%d %H:%M:%S', 
 errors = 'coerce')
X['End_Time'] = pd.to_datetime(X['End_Time'], 
 format = '%Y-%m-%d %H:%M:%S', 
 errors = 'coerce')


# In[23]:


# On créé une colonne pour chaque élément de nos dates de début d'accident

X['Start_Time_year'] = X['Start_Time'].dt.year
X['Start_Time_month'] = X['Start_Time'].dt.month
X['Start_Time_week'] = X['Start_Time'].dt.week
X['Start_Time_day'] = X['Start_Time'].dt.day
X['Start_Time_hour'] = X['Start_Time'].dt.hour
X['Start_Time_minute'] = X['Start_Time'].dt.minute
X['Start_Time_dayofweek'] = X['Start_Time'].dt.dayofweek


# In[24]:


# On créé une colonne pour chaque élément de nos dates de fin d'accident

X['End_Time_year'] = X['End_Time'].dt.year
X['End_Time_month'] = X['End_Time'].dt.month
X['End_Time_week'] = X['End_Time'].dt.week
X['End_Time_day'] = X['End_Time'].dt.day
X['End_Time_hour'] = X['End_Time'].dt.hour
X['End_Time_minute'] = X['End_Time'].dt.minute
X['End_Time_dayofweek'] = X['End_Time'].dt.dayofweek


# In[25]:


# Maintenant que l'on a créé nos colonnes, on supprime nos de base vue que l'on en a plus besoin

X = X.drop(['Start_Time'],axis=1)
X = X.drop(['End_Time'],axis=1)


# In[29]:


# On enlève toutes les valeurs NaN

X = X.dropna(how='any')


# In[30]:


X.shape, Y.shape


# In[31]:


X_tr, X_te, Y_tr, Y_te = TTS(X, Y,              # features, target
                            stratify = Y,       # Va prendre une proportion aux hasard de valeurs différentes histoire de ne pas avoir des cas où l'on a que des même valeur
                            random_state=777,   # Sert à fixer le harsard pour ne pas avoir des résultat différents à chaque tests.
                            train_size=0.8)     # 50% de X_train et Y_train et 50% de Y_test et Y_test


# In[60]:


knn = KNN(n_neighbors=21,
         weights='uniform',
         leaf_size=3)
knn.fit(X_tr, Y_tr)
train_preds = knn.predict(X_tr)
predictions = knn.predict(X_te)
knnTr = accuracy(train_preds, Y_tr)
knnTe= accuracy(predictions, Y_te)


# In[76]:


clf = RandomForestClassifier(random_state=0,      # Lecture aléatoire des données
                            n_estimators=21,      # Il va diviser X_tr en plusieurs arbres
                            max_depth=7)          # Il va spliter/augmenter la profondeur des arbres
clf.fit(X_tr, Y_tr)
train_preds = clf.predict(X_tr)
predictions = clf.predict(X_te)
clfTr = accuracy(train_preds, Y_tr)
clfTe= accuracy(predictions, Y_te)


# In[91]:


grad = GradientBoostingClassifier(random_state=60,        # Lecture aléatoire des données
                                 n_estimators=40,         # Nombre d'étape (modèle), plus il y a de modèle, plus il apprendra
                                 max_depth=10)            # Profondeur de l'arborescence
grad.fit(X_tr, Y_tr)
train_preds = grad.predict(X_tr)
predictions = grad.predict(X_te)
gradTr = accuracy(train_preds, Y_tr)
gradTe= accuracy(predictions, Y_te)


# In[ ]:


@app.get("/prediction/{algo}/")
async def prediction(algo: str):
    if algo == 'KNN':
        result = " Pour l\'algorithme %s, on à un score de %f d'accuracy en train et %f d'accuracy en test. " % (algo, knnTr, knnTe)
    elif algo == 'RF':
        result = " Pour l\'algorithme %s, on à un score de %f d'accuracy en train et %f d'accuracy en test. " % (algo, clfTr, clfTe)
    elif algo == 'GB':
        result = " Pour l\'algorithme %s, on à un score de %f d'accuracy en train et %f d'accuracy en test. " % (algo, gradTr, gradTe)
    else:
        result = " Vous devez choisir entre : KNN, RF (RandomForest) ou GB (GradientBoosting)"
    return result


# In[14]:


@app.get("/prediction/{algo}/{number}/")
async def prediction(algo: str, number: float):
    if number > 1 or number < 0:
        result = " Vous devez choisir entre : KNN, RF (RandomForest) ou GB (GradientBoosting) en plus du pourcentage de splitting entre le train et le test (train_size) entre 1 et 0."
    else:
        X_tr, X_te, Y_tr, Y_te = TTS(X, Y,              # features, target
                            stratify = Y,       # Va prendre une proportion aux hasard de valeurs différentes histoire de ne pas avoir des cas où l'on a que des même valeur
                            random_state=777,   # Sert à fixer le harsard pour ne pas avoir des résultat différents à chaque tests.
                            train_size=number)     # 50% de X_train et Y_train et 50% de Y_test et Y_test
    
    if algo == 'KNN':
        scoreTr, scoreTe = kn(number, X_tr, X_te, Y_tr, Y_te)
        result = " Pour l\'algorithme %s, on à un score de %f d'accuracy en train et %f d'accuracy en test. " % (algo, scoreTr, scoreTe)
    elif algo == 'RF':
        scoreTr, scoreTe = rf(number, X_tr, X_te, Y_tr, Y_te)
        result = " Pour l\'algorithme %s, on à un score de %f d'accuracy en train et %f d'accuracy en test. " % (algo, scoreTr, scoreTe)
    elif algo == 'GB':
        scoreTr, scoreTe = gb(number, X_tr, X_te, Y_tr, Y_te)
        result = " Pour l\'algorithme %s, on à un score de %f d'accuracy en train et %f d'accuracy en test. " % (algo, scoreTr, scoreTe)
    else:
        result = " Vous devez choisir entre : KNN, RF (RandomForest) ou GB (GradientBoosting) en plus du pourcentage de splitting entre le train et le test (train_size) entre 1 et 0."
    return result

