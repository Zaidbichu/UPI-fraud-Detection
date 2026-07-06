from FRAUD_DETECTION.exception.exception import FraudException
from FRAUD_DETECTION.logging.logger import logging
from FRAUD_DETECTION.entity.config_entity import (
    dataingestionconfig,
    datatransformationconfig,
    datavalidationconfig,
    trainingpipelineconfig,
    modeltrainingconfig
)
from FRAUD_DETECTION.entity.artifact_entity import (
    dataingestionartifact,
    Datatransformationartifact,
    Datavalidationartifact,
    Model_training_artifact
)
from FRAUD_DETECTION.COMPONENTS.data_ingestion import Dataingestion
from FRAUD_DETECTION.COMPONENTS.data_validation import Datavalidation
from FRAUD_DETECTION.COMPONENTS.data_tranformation import Datatransformation
from FRAUD_DETECTION.COMPONENTS.model_training import Modeltraining
import os,sys
class training_pipeline:
    def __init__(self):
        self.training_pipeline_config = None
        try:
            self.training_pipeline_config = trainingpipelineconfig()
        except Exception as e:
            raise FraudException(e,sys)
    def start_data_ingestion(self)->dataingestionartifact:
        try:
            logging.info("data ingestion has started")
            data_ingestion_config=dataingestionconfig(self.training_pipeline_config)
            data_ingestion=Dataingestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info("data ingestion has completed")
            return data_ingestion_artifact
        except Exception as e:
            raise FraudException(e,sys)
    def start_data_validation(self,data_ingestion_artifact:dataingestionartifact)->Datavalidationartifact:
        try:
            logging.info("we have enter the start data validation method")
            data_validation_config=datavalidationconfig(self.training_pipeline_config)
            data_validation=Datavalidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info("we have completed the data validation part")
            return data_validation_artifact
        except Exception as e:
            raise FraudException(e,sys)
    def start_data_transformation(self,data_validation_artifact:Datavalidationartifact)->Datatransformationartifact:
        try:
            logging.info("we have enter the transformation part")
            data_tranformation_config=datatransformationconfig(self.training_pipeline_config)
            data_transformation=Datatransformation(data_transformation_config=data_tranformation_config,data_validation_artifact=data_validation_artifact)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info('we have completed the data transformation part')
            return data_transformation_artifact
        except Exception as e:
            raise FraudException(e,sys)
    def start_model_training(self,data_transformation_artifact:Datatransformationartifact)->Model_training_artifact:
        try:
            logging.info("we have enter the model training part")
            model_training_config=modeltrainingconfig(self.training_pipeline_config)
            model_training=Modeltraining(model_training_config=model_training_config,data_transformation_artifact=data_transformation_artifact)
            model_training_artifact=model_training.initiate_model_training()
            logging.info("we have completed the final phase")
            return model_training_artifact
        except Exception as e:
            raise FraudException(e,sys)

    def run_pipeline(self)->Model_training_artifact:
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact)
            Model_training_artifact=self.start_model_training(data_transformation_artifact)
            return Model_training_artifact
        except Exception as e:
            raise FraudException(e,sys)
if __name__=="__main__":
    pipeline=training_pipeline()
    artifact=pipeline.run_pipeline()
    print(artifact)

    
