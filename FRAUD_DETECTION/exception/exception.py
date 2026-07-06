import os
import sys
from FRAUD_DETECTION.logging import logger
class FraudException(Exception):
    def __init__(self, error_message,error_details:sys):
            self.error_message=error_message
            _,_,exc_tb=error_details.exc_info()
            self.line_no=exc_tb.tb_lineno
            self.file_name=exc_tb.tb_frame.f_code.co_filename
    def __str__(self):
          return f"the error has been traced in the file [{self.file_name}] on the line no[{self.line_no}] and error details: {str(self.error_message)}"
