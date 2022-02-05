import datetime
import json
import logging
import os

import click
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

# common is a script to set parameters on production
try:
    import common  # noqa: F401
except ImportError:
    # ignore import error -> work anyway (but define logging)
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


def send_email_report(email_address: str, success: bool):
    """ Send an report email with the status to somebody """
    print("SEND EMAIL", os.environ.get("EMAIL_HOST"), success)


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
def success(
    output_dir: str,
    pipeline_type: str,
    run_id: str,
    execution_date: datetime.datetime,
    logical_date: datetime.datetime,
    headline: str,
    info: str,
    config: str,
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
    send_email_report("test@bluesquarehub.com", success=True)


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
def failure(
    output_dir: str,
    pipeline_type: str,
    run_id: str,
    execution_date: datetime.datetime,
    logical_date: datetime.datetime,
    headline: str,
    info: str,
    config: str,
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
    send_email_report("test@bluesquarehub.com", success=True)


if __name__ == "__main__":
    cli()
