import os

##defining the constant variables for training pipeline

target_column:str="isFraud"
pipeline_name:str="fraud_detection"
artifact_dir:str="artifact"
train_file_path:str="train.csv"
test_file_path:str="test.csv"
file_name:str="fraud_detection.csv"
schema_file_path=os.path.join("data_schema","schema.yaml")

##data ingestion constants
Data_ingestion_dir:str="data_ingestion"
Data_ingestion_train_test_split_ratio:float=0.2
Data_ingestion_raw_data_dir:str="raw_data"
Data_ingestion_ingested_dir:str="ingested_dir"

## data validation constants
Data_validation_dir:str="data_validation"
Data_validation_validated_dir:str="validated_dir"
Data_validation_invalid_dir:str="invalid_dir"
Data_validation_drift_report_dir:str="Drift_report"
Data_validation_drift_report_file_name:str="report.yaml"

##data transformation constans
Data_transformation_dir:str="data_transformation"
Data_transformation_transformed_dir:str="transformd"
Data_transform_processor_object_file_path:str="transformed_object"
Data_tranform_obj_file_name:str="processor.pkl"
Data_transformed_train_file_path:str='train.npy'
Data_transformed_test_file_path:str="test.npy"

##model training constants

model_training_dir:str="model_training"
model_training_file_name:str="model.pkl"
model_training_train_metric_dir:str="train_metric"
model_training_test_metric_dir:str="test_metric"
model_training_expected_score:float=0.7
model_training_overfitting_underfitting_threshold:float=0.05
