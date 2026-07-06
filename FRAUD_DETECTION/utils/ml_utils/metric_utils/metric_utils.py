from FRAUD_DETECTION.exception.exception import FraudException
from sklearn.metrics import f1_score,recall_score,precision_score
from FRAUD_DETECTION.logging.logger import logging
from FRAUD_DETECTION.entity.artifact_entity import ClassificationMetricArtifact
import sys
def get_classification_score(y_true,y_pred):
    logging.info("we have enterd in the classification metric")
    try:
        model_f1_score=f1_score(y_true,y_pred)
        model_precision_score=precision_score(y_true,y_pred)
        model_recall_score=recall_score(y_true,y_pred)
        classification_metric=ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric
    except Exception as e:
        raise FraudException(e,sys)