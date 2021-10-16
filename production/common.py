# - this file should be common to all docker images, to inject variables, change logger info etc on production
#   airflow environnement.
# - this file is in a separate directory because it's common to all pipelines. it's copied by the github workflow
# - pipelines with the file should work locally, with missing config etc. So please design it to gracefully degrade
#   instead of crashing everything.
# - pipelines should work without it. So please import it in pipelines in try/except ImportError.

from logging import getLogger

logger = getLogger(__name__)

logger.info("EXEC COMMON")
import os
logger.info("KEY ID", os.environ.get("AWS_ACCESS_KEY_ID"))
