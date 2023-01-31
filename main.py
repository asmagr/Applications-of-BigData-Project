import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random
import pickle

from pre_processing import *
from model_training import *
from predict import *

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    accu_score = accuracy_score(actual, pred)
    recall_scores = recall_score(actual, pred,average='macro')
    precision_scores = precision_score(actual, pred,average='macro')
    f1_scores = f1_score(actual, pred,average='weighted')
    return accu_score, recall_scores, precision_scores, f1_scores


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)


     
    
    
    df_train=application_train(num_rows = None, nan_as_category = False)
    train_df = df_train[df_train['TARGET'].notnull()]
    process_data=check_null_value_drop(train_df)
    
    
    labels = process_data['TARGET']
    features = process_data.drop(columns = ['TARGET', 'SK_ID_CURR'])
    train_features_name=list(features.columns)
    print('Train data size',features.shape)
    
    from sklearn.model_selection import KFold, train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=40, stratify=labels)
    
    num_estimators = random.randint(10, 50)
 
    min_samples_splt=random.randint(5, 25)
    
    
    # Test data without label only
    df_test=application_test_data(num_rows = None, nan_as_category = False)
    #train_df = df_train[df_train['TARGET'].notnull()]
    df_test_data=check_null_value_drop_test(df_test,train_features_name)
    
    print('Test data size',df_test_data.shape)
    


    with mlflow.start_run():
        model_rf=model_train(X_train, y_train,num_estimators,min_samples_splt)

        
        predicted_qualities=predict(X_test,model_rf)

        (accu_score, recall_scores, precision_scores, f1_scores) = eval_metrics(y_test, predicted_qualities)

        print("RF model (estimators={:f}, min_samples_split={:f}):".format(num_estimators, min_samples_splt))
        print("accuracy sccore: %s" % accu_score)
        print("Recall scores: %s" % recall_scores)
        print("precision scores: %s" % precision_scores)
        print("f1_scores: %s" % f1_scores)
        
        # Test data without label
        predicted_qualities_test=predict(df_test_data,model_rf)
        
        prediction_test_=pd.DataFrame(predicted_qualities_test)
        prediction_test_.to_csv('test_data_prediction.csv')
        

        # Store values in MLFLOW
        mlflow.log_param("number of estimators", num_estimators)
        mlflow.log_param("min samples split", min_samples_splt)
        mlflow.log_metric("accuracy sccore", accu_score)
        mlflow.log_metric("Recall score", recall_scores)
        mlflow.log_metric("precision score", precision_scores)
        mlflow.log_metric("f1 score", f1_scores)

        # Save model in local dic
        filename = 'finalized_model.sav'
        pickle.dump(model_rf, open(filename, 'wb'))
        
        # load the model
        #loaded_model = pickle.load(open(filename, 'rb'))
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model_rf, "model", registered_model_name="RF Model")
        else:
            mlflow.sklearn.log_model(model_rf, "model")
            
