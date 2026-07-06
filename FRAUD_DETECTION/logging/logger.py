import sys
import os
import logging
from datetime import datetime

log_file = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, log_file)

try:
    logging.basicConfig(
        filename=log_file_path,
        format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
except Exception:
    logging.basicConfig(
        stream=sys.stdout,
        format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

logger = logging.getLogger("fraud_detection")
