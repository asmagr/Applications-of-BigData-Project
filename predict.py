# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 22:37:02 2023

@author: A.Anjum
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

def predict(X_test,model_rf):
    predicted = model_rf.predict(X_test)
    return predicted