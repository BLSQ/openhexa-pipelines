from logging import getLogger

logger = getLogger(__name__)

logger.info("EXEC COMMON")
import os
logger.info("KEY ID", os.environ.get("AWS_ACCESS_KEY_ID"))
