import datetime
import json
import logging
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import click
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

# common is a script to set parameters on production
try:
    import common  # noqa: F401
except ImportError as e:
    # ignore import error -> work anyway (but define logging)
    print(f"Common code import error: {e}")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger(__name__)


class ReportError(Exception):
    """Report error."""


def generate_content(
    *,
    is_success: bool,
    headline: str,
    pipeline_type: str,
    run_id: str,
    execution_date: datetime.datetime,
    logical_date: datetime.datetime,
    info: str,
    config: str,
):
    if is_success:
        success_or_failure = "✅ This pipeline was run successfully."
    else:
        success_or_failure = "❌ This pipeline failed to execute properly."

    return f"""# OpenHexa Pipeline report
## {headline}
The content of this directory was created by a Pipeline run from OpenHexa.

{success_or_failure}

Key facts:
- This run comes from a pipeline of type **{pipeline_type}**
- The run has `{run_id}` as identifier (use this for troubleshooting purposes)
- The pipeline was run on **{execution_date.strftime("%B %d, %Y at %H:%M:%S")}**.
- The logical execution date of the pipeline is **{logical_date.strftime("%B %d, %Y at %H:%M:%S")}**.

## Additional info
{info}

## Used configuration
```
{json.dumps(json.loads(config), indent=2)}
```"""


def filesystem(target_path: str) -> AbstractFileSystem:
    """Guess filesystem based on path"""

    client_kwargs = {}
    if "://" in target_path:
        target_protocol = target_path.split("://")[0]
        if target_protocol == "s3":
            fs_class = S3FileSystem
            client_kwargs = {"endpoint_url": os.environ.get("AWS_S3_ENDPOINT")}
        elif target_protocol == "gcs":
            fs_class = GCSFileSystem
        elif target_protocol == "http" or target_protocol == "https":
            fs_class = HTTPFileSystem
        else:
            raise ValueError(f"Protocol {target_protocol} not supported.")
    else:
        fs_class = LocalFileSystem

    return fs_class(client_kwargs=client_kwargs)


PLAIN_MAIL_TEMPLATE = """
Hello Pipeline Owner,

The pipeline %(pipeline)s has finished its run.
Finish time: %(run_time)s UTC
Status: %(status_str)s

The report is available in S3.
"""

HTML_MAIL_TEMPLATE = """
<html>
    <body>
        <p>
            Hello Pipeline Owner,<br><br>
            The pipeline %(pipeline)s has finished its run.<br>
            Finish time: %(run_time)s UTC<br>
            Status: %(status_str)s<br><br>
            The report is available in S3.
        </p>
    </body>
</html>
"""


def send_email_report(email_address: str, pipeline: str, success: bool):
    """ Send an report email with the status to somebody """
    email_host = os.environ.get("EMAIL_HOST")
    email_port = int(os.environ.get("EMAIL_PORT"))
    email_host_user = os.environ.get("EMAIL_HOST_USER")
    email_host_password = os.environ.get("EMAIL_HOST_PASSWORD")
    email_pretty_from = os.environ.get("DEFAULT_FROM_EMAIL")

    if not (
        email_host
        and email_port
        and email_host_user
        and email_host_password
        and email_pretty_from
    ):
        logger.error("send_email_report(): missconfiguration, abort")
        return

    info = {
        "pipeline": pipeline,
        "run_time": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "status_str": "Success" if success else "Failure",
    }

    message = MIMEMultipart("alternative")
    message["Subject"] = f"Pipeline {pipeline} run: {info['status_str']}"
    message["From"] = email_pretty_from
    message["To"] = email_address
    message.attach(MIMEText(PLAIN_MAIL_TEMPLATE % info, "plain"))
    message.attach(MIMEText(HTML_MAIL_TEMPLATE % info, "html"))

    context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    with smtplib.SMTP_SSL(email_host, email_port, context=context) as server:
        server.login(email_host_user, email_host_password)
        server.sendmail(email_host_user, email_address, message.as_string())


@click.group()
def cli():
    pass


@cli.command()
@click.option("--output-dir", "-o", type=str, required=True, help="Output directory.")
@click.option(
    "--pipeline_type", "-t", type=str, required=True, help="ID of the DAG being run."
)
@click.option("--run_id", "-r", type=str, required=True, help="ID of the run itself.")
@click.option(
    "--execution_date",
    "-d",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"]),
    required=True,
    help="Execution date.",
)
@click.option(
    "--logical_date",
    "-l",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"]),
    required=True,
    help="Logical run date.",
)
@click.option(
    "--headline", "-h", type=str, required=True, help="Information about the DAG."
)
@click.option(
    "--info", "-i", type=str, required=True, help="Information about the DAG."
)
@click.option(
    "--config", "-c", type=str, required=True, help="Configuration used for the run."
)
@click.option(
    "--email_report",
    "-e",
    type=str,
    required=False,
    help="Send a copy of the report by email",
)
def success(
    output_dir: str,
    pipeline_type: str,
    run_id: str,
    execution_date: datetime.datetime,
    logical_date: datetime.datetime,
    headline: str,
    info: str,
    config: str,
    email_report: str,
):
    """Generates a success report for the pipeline run."""

    fs = filesystem(output_dir)
    fs.mkdirs(output_dir, exist_ok=True)

    output_file = f"{output_dir.rstrip('/')}/README.md"
    with fs.open(output_file, "w") as f:
        logger.debug(f"Writing README.md report file {output_file}.")
        content = generate_content(
            is_success=True,
            headline=headline,
            pipeline_type=pipeline_type,
            run_id=run_id,
            execution_date=execution_date,
            logical_date=logical_date,
            info=info,
            config=config,
        )
        f.write(content)
    if email_report:
        send_email_report(email_report, pipeline=headline, success=True)


@cli.command()
@click.option("--output-dir", "-o", type=str, required=True, help="Output directory.")
@click.option(
    "--pipeline_type", "-t", type=str, required=True, help="ID of the DAG being run."
)
@click.option("--run_id", "-r", type=str, required=True, help="ID of the run itself.")
@click.option(
    "--execution_date",
    "-d",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"]),
    required=True,
    help="Execution date.",
)
@click.option(
    "--logical_date",
    "-l",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"]),
    required=True,
    help="Logical run date.",
)
@click.option(
    "--headline", "-h", type=str, required=True, help="Information about the DAG."
)
@click.option(
    "--info", "-i", type=str, required=True, help="Information about the DAG."
)
@click.option(
    "--config", "-c", type=str, required=True, help="Configuration used for the run."
)
@click.option(
    "--email_report",
    "-e",
    type=str,
    required=False,
    help="Send a copy of the report by email",
)
def failure(
    output_dir: str,
    pipeline_type: str,
    run_id: str,
    execution_date: datetime.datetime,
    logical_date: datetime.datetime,
    headline: str,
    info: str,
    config: str,
    email_report: str,
):
    """Generates an error report for the pipeline run."""

    fs = filesystem(output_dir)
    fs.mkdirs(output_dir, exist_ok=True)

    output_file = f"{output_dir.rstrip('/')}/README.md"
    with fs.open(output_file, "w") as f:
        logger.debug(f"Writing README.md report file {output_file}.")
        content = generate_content(
            is_success=False,
            headline=headline,
            pipeline_type=pipeline_type,
            run_id=run_id,
            execution_date=execution_date,
            logical_date=logical_date,
            info=info,
            config=config,
        )
        f.write(content)
    if email_report:
        send_email_report(email_report, pipeline=headline, success=False)


if __name__ == "__main__":
    cli()
