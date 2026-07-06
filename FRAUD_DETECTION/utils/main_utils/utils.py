from FRAUD_DETECTION.exception.exception import FraudException
from FRAUD_DETECTION.logging.logger import logging
import numpy as np
import pandas as pd
import yaml,sys,os
import pickle
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import GridSearchCV
def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise FraudException(e,sys)
def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace and os.path.exist(file_path):
            os.remove(file_path)
        dir_path=os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w') as yaml_file:
            yaml.dump(content,yaml_file)
    except Exception as e:
        raise FraudException(e,sys)
def save_numpy_array(file_path:str,array:np.array)->None:
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb')as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise FraudException(e,sys)
def save_object(file_path:str,obj:object)->None:
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise FraudException(e,sys)
def load_numpy_array(file_path:str)->np.array:
    try:
        with open(file_path,'rb') as file:
            return np.load(file)
    except Exception as e:
        raise FraudException(e,sys)
def load_object(file_path:str):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise FraudException(e,sys)

def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)
            test_model_score = f1(y_test, y_test_pred)

            report[model_name] = test_model_score

            logging.info(f"{model_name} F1 score: {test_model_score}")

        return report

    except Exception as e:
        raise FraudException(e, sys)
        
            
 