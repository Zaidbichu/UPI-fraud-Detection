import numpy as np
import pandas as pd
import os,sys
from FRAUD_DETECTION.exception.exception import FraudException
from FRAUD_DETECTION.logging.logger import logging
from FRAUD_DETECTION.exception.exception import FraudException
from FRAUD_DETECTION.logging.logger import logger
from FRAUD_DETECTION.entity.config_entity import datatransformationconfig, trainingpipelineconfig,dataingestionconfig,datavalidationconfig
from FRAUD_DETECTION.entity.artifact_entity import Datavalidationartifact,Datatransformationartifact
from FRAUD_DETECTION.constants.training_pipeline import schema_file_path
from FRAUD_DETECTION.utils.main_utils.utils import read_yaml_file
from FRAUD_DETECTION.utils.main_utils.utils import save_numpy_array,save_object
from FRAUD_DETECTION.COMPONENTS.data_ingestion import Dataingestion
from FRAUD_DETECTION.COMPONENTS.data_validation import Datavalidation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
class Datatransformation:
    def __init__(self,data_validation_artifact:Datavalidationartifact,data_transformation_config:datatransformationconfig):
        self.data_transformation_config=data_transformation_config
        self.data_validation_artifact=data_validation_artifact
        self._schema_file_path=read_yaml_file(schema_file_path)
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FraudException(e,sys)
    def add_columns(self,df:pd.DataFrame):
        try:
            logging.info("we have enter the add columns method")
            df['diff_in_balance_sender_account']=df['oldbalanceOrg']-df['newbalanceOrig']
            df['receiver_balance_diff_account']=df['newbalanceDest']-df['oldbalanceDest']
            logging.info("the columns have been added succesfully")
            return df
        except Exception as e:
            raise FraudException(e,sys)

            
    def remove_the_columns(self,data:pd.DataFrame):
        try:
            logging.info("we have enter the drop column method")
            for col in data.columns:
                if col in self._schema_file_path['drop_columns']:
                    data.drop(col,axis=1,inplace=True)
            return data
        except Exception as e:
            raise FraudException(e,sys)
    def get_transformed_obj(self)->Pipeline:
        try:
            logging.info("we have enter the transformed object method")
            num_pipeline=Pipeline(
                steps=[
                    ("scalar",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            processor_obj=ColumnTransformer(
                transformers=[
                    ("num",num_pipeline,self._schema_file_path['numerical_columns']),
                    ("cat",cat_pipeline,self._schema_file_path['categorical_columns'])
                ]
            )
            return processor_obj
        except Exception as e:
            raise FraudException(e,sys)
    def initiate_data_transformation(self)->Datatransformationartifact:
        try:
            logging.info("we have enter in the data tranformation part")
            logging.info("we need to read the data")
            train_data=self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_data=self.read_data(self.data_validation_artifact.valid_test_file_path)
            logging.info("we have loaded the data succesfully")
            logging.info("we need to remove the unessecary columns now")
            new_train_data=self.remove_the_columns(train_data)
            new_test_data=self.remove_the_columns(test_data)
            logging.info("we have remove the unessecary columns")
            logging.info("we need to some important columns to it")
            new_train_data=self.add_columns(new_train_data)
            new_test_data=self.add_columns(new_test_data)
            input_feature_train_data=new_train_data.drop(columns=self._schema_file_path['target_column'])
            target_feature_train_data=new_train_data[self._schema_file_path['target_column']]

            input_feature_test_data=new_test_data.drop(columns=self._schema_file_path['target_column'])
            target_feature_test_data=new_test_data[self._schema_file_path['target_column']]

            processor_obj=self.get_transformed_obj()
            transformed_train_data=processor_obj.fit_transform(input_feature_train_data)
            transformed_test_data=processor_obj.transform(input_feature_test_data)

            train_arr=np.column_stack((transformed_train_data,np.array(target_feature_train_data)))
            test_arr=np.column_stack((transformed_test_data,np.array(target_feature_test_data)))
            save_numpy_array(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path,test_arr)
            save_object(self.data_transformation_config.prepocessor_obj_path,processor_obj)
            datatranforamtionartifact=Datatransformationartifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_obj_file_path=self.data_transformation_config.prepocessor_obj_path
            )
            return datatranforamtionartifact
        except Exception as e:
            raise FraudException(e,sys)
if __name__ == "__main__":
        try:
            training_pipeline_config = trainingpipelineconfig()

            data_ingestion_config = dataingestionconfig(training_pipeline_config)
            data_ingestion = Dataingestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            data_validation_config = datavalidationconfig(training_pipeline_config)
            data_validation = Datavalidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()

            data_transformation_config = datatransformationconfig(training_pipeline_config)
            data_transformation = Datatransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            print(data_transformation_artifact)

        except Exception as e:
            raise FraudException(e, sys)