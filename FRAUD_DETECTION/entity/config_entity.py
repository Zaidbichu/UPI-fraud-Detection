import os
import datetime
from FRAUD_DETECTION.constants import training_pipeline
class trainingpipelineconfig:
    def __init__(self):
        timestamp=datetime.datetime.now().strftime('%m_%d_%y_%H_%M_%S')
        self.pipeline_name=training_pipeline.pipeline_name
        self.artifact_name=training_pipeline.artifact_dir
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.timestamp:str=timestamp
class dataingestionconfig:
    def __init__(self,training_pipeline_config:trainingpipelineconfig):
        self.data_ingestion_dir:str=os.path.join(training_pipeline_config.artifact_dir,training_pipeline.Data_ingestion_dir)
        self.raw_data_file_path:str=os.path.join(self.data_ingestion_dir,training_pipeline.Data_ingestion_raw_data_dir,training_pipeline.file_name)
        self.train_file_path:str=os.path.join(self.data_ingestion_dir,training_pipeline.Data_ingestion_ingested_dir,training_pipeline.train_file_path)
        self.test_file_path:str=os.path.join(self.data_ingestion_dir,training_pipeline.Data_ingestion_ingested_dir,training_pipeline.test_file_path)
        self.train_test_split_ratio:float=training_pipeline.Data_ingestion_train_test_split_ratio
        
class datavalidationconfig:
    def __init__(self,training_pipeline_config:trainingpipelineconfig):
        self.data_validation_dir:str=os.path.join(training_pipeline_config.artifact_dir,training_pipeline.Data_validation_dir) 
        self.valid_train_file_path:str=os.path.join(self.data_validation_dir,training_pipeline.Data_validation_validated_dir,training_pipeline.train_file_path)   
        self.valid_test_file_path:str=os.path.join(self.data_validation_dir,training_pipeline.Data_validation_validated_dir,training_pipeline.test_file_path)
        self.invalid_train_file_path:str=os.path.join(self.data_validation_dir,training_pipeline.Data_validation_invalid_dir,training_pipeline.train_file_path)
        self.invalid_test_file_path:str=os.path.join(self.data_validation_dir,training_pipeline.Data_validation_invalid_dir,training_pipeline.test_file_path)
        self.drift_report_file_path:str=os.path.join(
            self.data_validation_dir,training_pipeline.Data_validation_drift_report_dir,training_pipeline.Data_validation_drift_report_file_name
        )
class datatransformationconfig:
    def __init__(self,training_pipeline_config:trainingpipelineconfig):
        self.data_tranformed_dir:str=os.path.join(training_pipeline_config.artifact_dir,training_pipeline.Data_transformation_dir)
        self.transformed_train_file_path:str=os.path.join(self.data_tranformed_dir,training_pipeline.Data_transformation_transformed_dir,training_pipeline.Data_transformed_train_file_path)
        self.transformed_test_file_path:str=os.path.join(self.data_tranformed_dir,training_pipeline.Data_transformation_transformed_dir,training_pipeline.Data_transformed_test_file_path)
        self.prepocessor_obj_path:str=os.path.join(self.data_tranformed_dir,training_pipeline.Data_transform_processor_object_file_path,training_pipeline.Data_tranform_obj_file_name)

class modeltrainingconfig:
    def __init__(self,training_pipeline_config:trainingpipelineconfig):
        self.model_training_dir=os.path.join(training_pipeline_config.artifact_dir,training_pipeline.model_training_dir)
        self.train_metric_file_path:str=os.path.join(self.model_training_dir,training_pipeline.model_training_train_metric_dir)
        self.test_metric_file_path:str=os.path.join(self.model_training_dir,training_pipeline.model_training_train_metric_dir)
        self.save_model_file_path:str=os.path.join(
            self.model_training_dir,training_pipeline.model_training_file_name
        )

