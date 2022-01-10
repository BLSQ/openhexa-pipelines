import datetime
import json

from click.testing import CliRunner

import report


def test_success():
    runner = CliRunner()
    sample_date_1 = datetime.datetime.now(tz=datetime.timezone.utc).replace(
        microsecond=0
    )
    sample_date_2 = datetime.datetime.now(tz=datetime.timezone.utc)
    result = runner.invoke(
        report.cli,
        [
            "success",
            "-o",
            "/code/some_dir",
            "-t",
            "Airflow",
            "-r",
            "xb22",
            "-d",
            sample_date_1.isoformat(),
            "-l",
            sample_date_2.isoformat(),
            "-h",
            "Amazing pipeline",
            "-i",
            "This pipeline does wonders!",
            "-c",
            json.dumps({"foo": "bar"}),
        ],
    )
    assert result.exit_code == 0

    with open("/code/some_dir/README.md") as f:
        actual_content = f.read()
        expected_content = f"""# OpenHexa Pipeline report
## Amazing pipeline
The content of this directory was created by a Pipeline run from OpenHexa.

✅ This pipeline was run successfully.

Key facts:
- This run comes from a pipeline of type **Airflow**
- The run has `xb22` as identifier (use this for troubleshooting purposes)
- The pipeline was run on **{sample_date_1.strftime("%B %d, %Y at %H:%M:%S")}**.
- The logical execution date of the pipeline is **{sample_date_2.strftime("%B %d, %Y at %H:%M:%S")}**.

## Additional info
This pipeline does wonders!

## Used configuration
```
{{
  "foo": "bar"
}}
```"""

    assert actual_content == expected_content


def test_failure():
    runner = CliRunner()
    sample_date_1 = datetime.datetime.now(tz=datetime.timezone.utc).replace(
        microsecond=0
    )
    sample_date_2 = datetime.datetime.now(tz=datetime.timezone.utc)
    result = runner.invoke(
        report.cli,
        [
            "failure",
            "-o",
            "/code/some_dir",
            "-t",
            "Airflow",
            "-r",
            "xb22",
            "-d",
            sample_date_1.isoformat(),
            "-l",
            sample_date_2.isoformat(),
            "-h",
            "Amazing pipeline",
            "-i",
            "This pipeline does wonders!",
            "-c",
            json.dumps({"foo": "bar"}),
        ],
    )
    assert result.exit_code == 0

    with open("/code/some_dir/README.md") as f:
        actual_content = f.read()
        expected_content = f"""# OpenHexa Pipeline report
## Amazing pipeline
The content of this directory was created by a Pipeline run from OpenHexa.

❌ This pipeline failed to execute properly.

Key facts:
- This run comes from a pipeline of type **Airflow**
- The run has `xb22` as identifier (use this for troubleshooting purposes)
- The pipeline was run on **{sample_date_1.strftime("%B %d, %Y at %H:%M:%S")}**.
- The logical execution date of the pipeline is **{sample_date_2.strftime("%B %d, %Y at %H:%M:%S")}**.

## Additional info
This pipeline does wonders!

## Used configuration
```
{{
  "foo": "bar"
}}
```"""

    assert actual_content == expected_content
