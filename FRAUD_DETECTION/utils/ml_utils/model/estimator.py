from FRAUD_DETECTION.exception.exception import FraudException
from FRAUD_DETECTION.logging.logger import logging
import os,sys
class fraud_model:
    def __init__(self,model,processor):
       
        try:
           self.model=model
           self.processor=processor
        except Exception as e:
           raise FraudException(e,sys)
    def predict(self,data):
        try:
            x_transform=self.processor.transform(data)
            y_predict=self.model.predict(x_transform)
            return y_predict

        except Exception as e:
            raise FraudException(e,sys)
    def predict_proba(self,data):
        try:
            x_transform=self.processor.transform(data)
            y_pro=self.model.predict_proba(x_transform)
            return y_pro
        except Exception as e:
            raise FraudException(e,sys)
