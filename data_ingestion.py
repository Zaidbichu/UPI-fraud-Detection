import numpy as np
import pandas as pd
import os,sys
import sklearn

from FRAUD_DETECTION.entity.config_entity import dataingestionconfig, trainingpipelineconfig
from FRAUD_DETECTION.entity.artifact_entity import dataingestionartifact
from FRAUD_DETECTION.exception.exception import FraudException
from FRAUD_DETECTION.logging.logger import logging
from sklearn.model_selection import train_test_split
class Dataingestion:
    def __init__(self,data_ingestion_config:dataingestionconfig):
        self.data_ingestion_config=data_ingestion_config
    def Load_data(self):
        try:
            logging.info("we have now enter in the load data method")
            logging.info("we will now load the data")
            df=pd.read_csv("upi_dataset/upi fraud dataset.csv")
            logging.info("we have loaded the data ")
            return df
        except Exception as e:
            raise FraudException(e,sys)
    def export_data_to_feature_store(self,df:pd.DataFrame):
        try:
            logging.info("we have enter to the feature store method")
            feature_store=self.data_ingestion_config.raw_data_file_path
            dir_path=os.path.dirname(feature_store)
            os.makedirs(dir_path,exist_ok=True)
            df.to_csv(feature_store,index=False,header=True)
            logging.info("we have exported the data to feature store")
            return df
        except Exception as e:
            raise FraudException(e,sys)
    def split_data_as_train_test(self,df:pd.DataFrame):
        try:
            logging.info("we have enter the train and test spli method")
            train,test=train_test_split(df,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            logging.info("we have now splitted the data to train and test")
            dir_path=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            train.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)
            logging.info("we have exported the data to train and test files")
        except Exception as e:
            raise FraudException(e,sys)

    def initiate_data_ingestion(self)->dataingestionartifact:
        try:
            logging.info("we have entered in the inititate data ingetion method")
            logging.info("we are loading the data from database")
            dataset=self.Load_data()
            logging.info("we have loaded the data as dataframe")
            logging.info("we are now exporting the data to feature store")
            dataset=self.export_data_to_feature_store(dataset)
            logging.info("we have exported the data to feature store")
            logging.info("now the data is ready to be splitted in train and test")
            self.split_data_as_train_test(dataset)
            data_ingestion_artifact=dataingestionartifact(raw_data_path=self.data_ingestion_config.raw_data_file_path,train_file_path=self.data_ingestion_config.train_file_path,test_file_path=self.data_ingestion_config.test_file_path)
            logging.info("we have created the data ingestion artifact")
            return data_ingestion_artifact
        except Exception as e:
            raise FraudException(e,sys)
