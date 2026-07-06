import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os,sys
from FRAUD_DETECTION.exception.exception import FraudException
from FRAUD_DETECTION.logging.logger import logging
from FRAUD_DETECTION.entity.config_entity import modeltrainingconfig
from FRAUD_DETECTION.entity.artifact_entity import Model_training_artifact,Datatransformationartifact
from FRAUD_DETECTION.utils.main_utils.utils import load_numpy_array,evaluate_model,load_object,save_object
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import(
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from FRAUD_DETECTION.utils.ml_utils.model.estimator import fraud_model
from FRAUD_DETECTION.utils.ml_utils.metric_utils.metric_utils import get_classification_score

class Modeltraining:
    def __init__(self,model_training_config:modeltrainingconfig,data_transformation_artifact:Datatransformationartifact):
        self.data_transformation_artifact=data_transformation_artifact
        self.model_training_config=model_training_config
    def model_train(self,x_train,y_train,x_test,y_test):
        try:
            logging.info('we have enter the model training part')
          
            models = {
                    "decision_tree_classifier": DecisionTreeClassifier(
                     max_depth=15,
                     class_weight="balanced",
                     random_state=42
            ),
                "random_forest_classifier": RandomForestClassifier(
                    n_estimators=50,
                    max_depth=15,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42
                )
            }
            '''
            params = {
                    "Random_forest_classifier": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    },
                    "decision_tree_classifier": {
                        "criterion": ["gini", "entropy", "log_loss"],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    },
                    "k_neighbour_classifier": {
                        "n_neighbors": [3, 5, 7, 9],
                        "weights": ["uniform", "distance"],
                        "metric": ["minkowski", "euclidean", "manhattan"]
                    },
                    "Ada_boost_classifier": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1.0]
                    },
                    "gradient_boost_classifier": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 10],
                        "subsample": [0.8, 1.0]
                    }
                }'''
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_score=max(model_report.values())
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]
            best_model=models[best_model_name]
            ''' for model_name, score in model_report.items():
                    print(f"{model_name}: {score}")'''
            best_model.fit(x_train, y_train)
            y_train_pred=best_model.predict(x_train)
            classification_trained_metric=get_classification_score(y_train,y_train_pred)
            Y_test_pred=best_model.predict(x_test)
            classfication_test_metric=get_classification_score(y_test,Y_test_pred)
            processor=load_object(self.data_transformation_artifact.preprocessor_obj_file_path)
            fraud_model_instance=fraud_model(best_model,processor)
            save_object(self.model_training_config.save_model_file_path,obj=fraud_model_instance)
            
        
        
            
            model_artifact = Model_training_artifact(
                model_train_file_path=self.model_training_config.save_model_file_path,
                train_metric_artifact=classification_trained_metric,
                test_metric_artifact=classfication_test_metric

            )
            logging.info(f"model artifact completed {model_artifact}")
            return model_artifact
        except Exception as e:
            raise FraudException(e,sys)
        




            


    def initiate_model_training(self)->Model_training_artifact:
        try:
            logging.info("we have enter the model training part")
            logging.info("we will now load the data")
            train_data=load_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_data=load_numpy_array(self.data_transformation_artifact.transformed_test_file_path)
            x_train,y_train,x_test,y_test=(
                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1]
            )
            model_train_artifact=self.model_train(x_train,y_train,x_test,y_test)
            return model_train_artifact
        except Exception as e:
            raise FraudException(e,sys)
