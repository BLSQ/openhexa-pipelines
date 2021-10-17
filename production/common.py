# - this file should be common to all docker images, to inject variables, change logger info etc on production
#   airflow environnement.
# - this file is in a separate directory because it's common to all pipelines. it's copied by the github workflow
# - pipelines with the file should work locally, with missing config etc. So please design it to gracefully degrade
#   instead of crashing everything.
# - pipelines should work without it. So please import it in pipelines in try/except ImportError.

import logging
import logging.config
import os

logger = logging.getLogger(__name__)
logger.info("Execute common")

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {"format": "%(asctime)s %(levelname)s: %(message)s"},
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
)

if "SENTRY_DSN" in os.environ:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration

    # inject sentry into logging config. set level to ERROR, we don't really want the rest?
    sentry_logging = LoggingIntegration(level=logging.ERROR, event_level=logging.ERROR)
    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        integrations=[sentry_logging],
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "1")),
        send_default_pii=True,
        environment=os.environ.get("SENTRY_ENVIRONMENT"),
    )
