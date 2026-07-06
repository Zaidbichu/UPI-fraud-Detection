from dataclasses import dataclass
@dataclass
class dataingestionartifact:
    raw_data_path:str
    train_file_path:str
    test_file_path:str
@dataclass
class Datavalidationartifact:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str
@dataclass
class Datatransformationartifact:
    preprocessor_obj_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str
@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
@dataclass
class Model_training_artifact:
    model_train_file_path:str
    train_metric_artifact:ClassificationMetricArtifact
    test_metric_artifact:ClassificationMetricArtifact

    
