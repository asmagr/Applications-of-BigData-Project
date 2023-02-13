
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def model_train(X_train, y_train,num_estimators,min_samples_splt):
    model_rf = RandomForestClassifier(n_estimators=num_estimators,
                                      criterion='entropy',
     min_samples_split= min_samples_splt,
     min_samples_leaf= 2,
     max_features='auto',
     max_depth= None,
     bootstrap=False)
    
    model_rf.fit(X_train, y_train)
    return model_rf


