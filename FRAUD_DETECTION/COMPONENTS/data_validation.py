from FRAUD_DETECTION.exception.exception import FraudException
from FRAUD_DETECTION.logging.logger import logging
from FRAUD_DETECTION.COMPONENTS.data_ingestion import Dataingestion
from FRAUD_DETECTION.entity.config_entity import datavalidationconfig,dataingestionconfig,trainingpipelineconfig
from FRAUD_DETECTION.entity.artifact_entity import dataingestionartifact,Datavalidationartifact
from FRAUD_DETECTION.constants.training_pipeline import schema_file_path
from FRAUD_DETECTION.utils.main_utils.utils import read_yaml_file,write_yaml_file
import os,sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class Datavalidation:
    def __init__(self,data_ingestion_artifact:dataingestionartifact,data_validation_config:datavalidationconfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_file_path=read_yaml_file(schema_file_path)
        except Exception as e:
            raise FraudException(e,sys)
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FraudException(e,sys)
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=dataframe.shape[1]
            print("number of columns in the dataframe is {}".format(number_of_columns))
            if number_of_columns==len(self._schema_file_path["columns"]):
                return True
            else:
                return False
        except Exception as e:
            raise FraudException(e,sys)
    def missing_columns(self,df:pd.DataFrame)->bool:
        try:
            missing_columns=[]
            for col in self._schema_file_path['columns'].keys():
                if col not in df.columns:
                        missing_columns.append(col)
            if len(missing_columns)==0:
                return True
            return False
        except Exception as e:
            raise FraudException(e,sys)
    def detect_drift_dataset(self,base_df,current_df,threshold=0.5):
        try:
            logging.info("we have enterd into the detect drift dataset")
            status=True
            report={}
            for col in base_df.columns:
                df1=base_df[col]
                df2=current_df[col]
                result=ks_2samp(df1,df2)
                if result.pvalue<threshold:
                    is_found=True
                    status=False
                else:
                    is_found=False
                report[col]={
                    "p_value":float(result.pvalue),
                    "drift_status":is_found
                }
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            dir_name=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_name,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
            return status
        except Exception as e:
            raise FraudException(e,sys)

    def initiate_data_validation(self)->Datavalidationartifact:
        try:
            logging.info("we have enter in the data validation method")
            logging.info("we are reading the train and test file for validation")
            train_dataframe=self.read_data(self.data_ingestion_artifact.train_file_path)
            test_dataframe=self.read_data(self.data_ingestion_artifact.test_file_path)
            status=self.validate_number_of_columns(train_dataframe)
            if not status:
                logging.info("train file is validated does not have the same number of columns as schema file")
            else:
                logging.info("train file is validated and have the same number of columns as schema file")
                
            test_status=self.validate_number_of_columns(test_dataframe)
            if not test_status:
                logging.info("test file is validated does not have the same number of columns as schema file")
            else:
                logging.info("test file is validated and have the same number of columns as schema file")
            missing_column_train=self.missing_columns(train_dataframe)
            if not missing_column_train:
                logging.info("missing columns are present in train dataset")
            else:
                logging.info("there are no missing columns in the dataset")
            missing_column_test=self.missing_columns(test_dataframe)
            if not missing_column_test:
                logging.info("there are missing columns in the test dataset")
            else:
                logging.info("missing columns are not present in the test dataset")
            stauts=self.detect_drift_dataset(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)
            datavalidationartifact=Datavalidationartifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path

            )
            return datavalidationartifact
        except Exception as e:
            raise FraudException(e,sys)
if __name__=="__main__":
    try:
        training_pipeline_config=trainingpipelineconfig()
        data_ingestion_config=dataingestionconfig(training_pipeline_config=training_pipeline_config)
        data_ingestion=Dataingestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        data_validation_config=datavalidationconfig(training_pipeline_config=training_pipeline_config)
        data_validation=Datavalidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
        data_validation.initiate_data_validation()
    except Exception as e:
        raise FraudException(e,sys)
